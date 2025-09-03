import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from typing import Dict, Optional, List
from pathlib import Path
import fitz  # PyMuPDF
from .slip_rule_parser import RuleBasedSlipParser
from .llm_parser import LLMSlipParser

# Optional PaddleOCR import (graceful fallback if unavailable)
try:
    from paddleocr import PaddleOCR  # type: ignore
    _paddle_available = True
except Exception:
    _paddle_available = False

class OCRProcessor:
    def __init__(self):
        # If Tesseract is not in PATH, set the executable path like below (Windows example):
        # pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        self.amount_patterns = [
            r'amount[:\s]*[\$]?([0-9,]+\.?[0-9]*)',
            r'total[:\s]*[\$]?([0-9,]+\.?[0-9]*)',
            r'deposit[:\s]*[\$]?([0-9,]+\.?[0-9]*)',
            r'[\$]([0-9,]+\.[0-9]{2})',
        ]

        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}',
        ]

        self.bank_patterns = {
            'SBI': r'state\s+bank\s+of\s+india|sbi',
            'HDFC': r'hdfc\s+bank|hdfc',
            'ICICI': r'icici\s+bank|icici',
            'AXIS': r'axis\s+bank|axis',
            'PNB': r'punjab\s+national\s+bank|pnb',
            'DCB': r'dcb\s*bank',
        }
        self.rule_parser = RuleBasedSlipParser()
        self.llm_parser = LLMSlipParser()
        self.paddle_ocr = None
        if _paddle_available:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("🧠 PaddleOCR initialized")
            except Exception:
                self.paddle_ocr = None

    # --- Image preprocessing helpers --------------------------------------------------------
    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """Guarantee a 3-channel BGR image for OCR engines that expect color input."""
        if image is None:
            return image
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        height, width = processed.shape
        if height < 300:
            scale = 300 / height
            new_width = int(width * scale)
            processed = cv2.resize(processed, (new_width, 300), interpolation=cv2.INTER_CUBIC)

        return processed

    def preprocess_cv_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        height, width = processed.shape
        if height < 300:
            scale = 300 / height
            new_width = int(width * scale)
            processed = cv2.resize(processed, (new_width, 300), interpolation=cv2.INTER_CUBIC)

        return processed

    def _generate_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple processed variants to increase OCR robustness."""
        variants: List[np.ndarray] = []
        bgr = self._ensure_bgr(image)
        variants.append(bgr)

        # Grayscale and CLAHE
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # Global Otsu
        _, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholds
        ada_mean = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
        ada_gauss = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

        # Inversions
        inv_otsu = cv2.bitwise_not(otsu)
        inv_mean = cv2.bitwise_not(ada_mean)
        inv_gauss = cv2.bitwise_not(ada_gauss)

        # Morph tweaks
        kernel = np.ones((1, 1), np.uint8)
        close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

        # Upscaled versions (helps Tesseract on small text)
        h, w = gray.shape
        if max(h, w) < 600:
            scale = 4.0
        elif max(h, w) < 1000:
            scale = 3.0
        elif max(h, w) < 1600:
            scale = 2.0
        else:
            scale = 1.5
        up_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        up_otsu = cv2.resize(close, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Unsharp masking for clarity
        gauss = cv2.GaussianBlur(up_bgr, (0, 0), sigmaX=1.0)
        sharp_up_bgr = cv2.addWeighted(up_bgr, 1.5, gauss, -0.5, 0)

        # Deskewed versions based on dominant line angles
        def _deskew(src_gray: np.ndarray) -> np.ndarray:
            try:
                edges = cv2.Canny(src_gray, 50, 150)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
                if lines is None:
                    return src_gray
                angles = []
                for l in lines[:100]:
                    rho, theta = l[0]
                    deg = (theta * 180.0 / np.pi) - 90.0
                    if -45 <= deg <= 45:
                        angles.append(deg)
                if not angles:
                    return src_gray
                angle = float(np.median(angles))
                hh, ww = src_gray.shape[:2]
                M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1.0)
                return cv2.warpAffine(src_gray, M, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            except Exception:
                return src_gray

        deskew_otsu = _deskew(otsu)
        deskew_gray = _deskew(gray_clahe)

        # Convert single-channel to BGR where appropriate
        for cand in [gray_clahe, otsu, ada_mean, ada_gauss, inv_otsu, inv_mean, inv_gauss, close, up_otsu, deskew_otsu, deskew_gray]:
            variants.append(self._ensure_bgr(cand))
        variants.append(up_bgr)
        variants.append(sharp_up_bgr)

        return variants

    def _maybe_split_halves(self, image: np.ndarray) -> List[np.ndarray]:
        """Split very wide images into left/right halves to isolate a single slip."""
        bgr = self._ensure_bgr(image)
        h, w = bgr.shape[:2]
        if w / max(h, 1) >= 1.6:  # likely two slips side-by-side
            mid = w // 2
            left = bgr[:, :mid]
            right = bgr[:, mid:]
            return [left, right]
        return [bgr]

    def render_pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")
                np_buffer = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                if image is not None:
                    images.append(image)
        finally:
            doc.close()
        return images

    def extract_text(self, image: np.ndarray) -> str:
        """Run OCR with several configs and optionally PaddleOCR, pick the longest output."""
        configs = [
            '-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-/: ',
            '-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-/: ',
            '-l eng --oem 1 --psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-/: ',
            '-l eng --oem 3 --psm 4',
            '-l eng --oem 1 --psm 12'
        ]
        best_text = ''
        # Tesseract on the given image
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(image, config=cfg).strip()
                if len(txt) > len(best_text):
                    best_text = txt
            except Exception:
                continue

        # Optional PaddleOCR fallback - choose longer text
        if self.paddle_ocr is not None:
            try:
                img_for_paddle = self._ensure_bgr(image)
                result = self.paddle_ocr.ocr(img_for_paddle, cls=True)
                paddle_lines: List[str] = []
                for res in result:
                    for line in res:
                        try:
                            paddle_lines.append(str(line[1][0]))
                        except Exception:
                            pass
                paddle_text = '\n'.join(paddle_lines).strip()
                if len(paddle_text) > len(best_text):
                    best_text = paddle_text
            except Exception:
                pass

        return best_text

    def _extract_best_from_variants(self, images: List[np.ndarray]) -> str:
        """Run OCR across multiple image variants and return the most informative text."""
        best_text = ''
        for img in images:
            text = self.extract_text(img)
            if len(text) > len(best_text):
                best_text = text
        return best_text

    def extract_amount(self, text: str) -> Optional[float]:
        if not text:
            return None
        text_lower = text.lower()

        def normalize_digits_like(s: str) -> str:
            table = str.maketrans({
                'o': '0', 'O': '0',
                'l': '1', 'i': '1', 'I': '1', '|': '1',
                's': '5', 'S': '5',
                'b': '8', 'B': '8',
                'g': '6', 'G': '6',
                'z': '2', 'Z': '2',
                'q': '9', 'Q': '9',
                'j': '3', 'J': '3',
                'v': '7', 'V': '7',
                '—': '-', '–': '-', '−': '-',
            })
            return s.translate(table)

        normalized = normalize_digits_like(text)
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 0.01 <= amount <= 10000000:
                        return amount
                except (ValueError, IndexError):
                    continue
            # Try again on normalized text for OCR confusions (O->0, l->1, etc.)
            matches2 = re.finditer(pattern, normalized, re.IGNORECASE)
            for match in matches2:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount_str = re.sub(r"[^0-9.]+", "", amount_str)
                    amount = float(amount_str)
                    if 0.01 <= amount <= 10000000:
                        return amount
                except (ValueError, IndexError):
                    continue
        return None

    def extract_date(self, text: str) -> Optional[str]:
        if not text:
            return None
        for pattern in self.date_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches.group(1)
                return self.standardize_date(date_str)
        return None

    def standardize_date(self, date_str: str) -> str:
        from datetime import datetime
        formats = [
            '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
            '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',
            '%m/%d/%y', '%d/%m/%y', '%y/%m/%d',
            '%m-%d-%y', '%d-%m-%y', '%y-%m-%d',
        ]
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str

    def extract_bank_name(self, text: str) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()
        for bank, pattern in self.bank_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return bank
        return None

    def extract_account_number(self, text: str) -> Optional[str]:
        if not text:
            return None
        # Normalize common OCR confusions to digits
        def _normalize_digits(s: str) -> str:
            table = str.maketrans({
                'o': '0', 'O': '0',
                'l': '1', 'i': '1', 'I': '1', '|': '1',
                's': '5', 'S': '5',
                'b': '8', 'B': '8',
                'g': '6', 'G': '6',
                'z': '2', 'Z': '2',
                'q': '9', 'Q': '9',
            })
            return s.translate(table)

        patterns = [
            r'a/?c\s*(?:no\.?|number)?\s*[:\-]?\s*(\w{6,20})',
            r'account\s*(?:no\.?|number)?\s*[:\-]?\s*(\w{6,20})',
            r'acc\s*no\.?\s*[:\-]?\s*(\w{6,20})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = _normalize_digits(match.group(1))
                digits = re.sub(r"\D", "", candidate)
                if 6 <= len(digits) <= 20:
                    return digits
        return None

    def calculate_confidence(self, extracted_data: Dict) -> float:
        confidence = 0.0
        if extracted_data.get('amount'):
            confidence += 0.4
        if extracted_data.get('date'):
            confidence += 0.3
        if extracted_data.get('bank_name'):
            confidence += 0.2
        if extracted_data.get('account_number'):
            confidence += 0.1
        return confidence

    async def process_deposit_slip(self, image_path: str, mode: str = "hybrid") -> Dict:
        print(f"🔍 OCRProcessor.process_deposit_slip: Starting with mode={mode}, file={image_path}")
        try:
            lower_path = image_path.lower()
            if lower_path.endswith('.pdf'):
                print(f"📄 PDF DETECTED: Processing PDF file")
                page_images = self.render_pdf_to_images(image_path)
                if not page_images:
                    raise ValueError("Could not render any pages from PDF")
                print(f"📄 PDF RENDERED: {len(page_images)} pages")

                best_data: Optional[Dict] = None
                for i, page_image in enumerate(page_images):
                    print(f"📄 PROCESSING PAGE {i+1}/{len(page_images)}")
                    # Try page, its halves, and multiple preprocess variants
                    candidates: List[np.ndarray] = []
                    for sub in self._maybe_split_halves(page_image):
                        candidates.extend(self._generate_variants(sub))
                    text = self._extract_best_from_variants(candidates)
                    print(f"📝 EXTRACTED TEXT (page {i+1}): {text[:100]}...")
                    if mode == "ocr":
                        print(f"🔍 USING OCR MODE")
                        extracted_data = {
                            'amount': self.extract_amount(text),
                            'date': self.extract_date(text),
                            'bank_name': self.extract_bank_name(text),
                            'account_number': self.extract_account_number(text),
                            'raw_text': text,
                            'confidence': 0.0,
                            'processing_details': {
                                'mode_used': 'ocr',
                                'amount_source': 'ocr' if self.extract_amount(text) else None,
                                'date_source': 'ocr' if self.extract_date(text) else None,
                                'bank_source': 'ocr' if self.extract_bank_name(text) else None,
                                'account_source': 'ocr' if self.extract_account_number(text) else None,
                                'ocr_confidence': self.calculate_confidence(extracted_data)
                            }
                        }
                        extracted_data['confidence'] = self.calculate_confidence(extracted_data)
                    elif mode == "llm":
                        print(f"🤖 USING LLM MODE")
                        llm_data, llm_conf = await self.llm_parser.parse(text)
                        print(f"🤖 LLM RESULT: {llm_data}, confidence: {llm_conf}")
                        extracted_data = {
                            'amount': llm_data.get('amount'),
                            'date': llm_data.get('date'),
                            'bank_name': llm_data.get('bank_name'),
                            'account_number': llm_data.get('account_number'),
                            'raw_text': text,
                            'confidence': llm_conf,
                            'processing_details': {
                                'mode_used': 'llm',
                                'amount_source': 'llm' if llm_data.get('amount') else None,
                                'date_source': 'llm' if llm_data.get('date') else None,
                                'bank_source': 'llm' if llm_data.get('bank_name') else None,
                                'account_source': 'llm' if llm_data.get('account_number') else None,
                                'llm_confidence': llm_conf
                            }
                        }
                    else:
                        extracted_data = {
                            'amount': self.extract_amount(text),
                            'date': self.extract_date(text),
                            'bank_name': self.extract_bank_name(text),
                            'account_number': self.extract_account_number(text),
                            'raw_text': text,
                            'confidence': 0.0
                        }
                        extracted_data['confidence'] = self.calculate_confidence(extracted_data)
                    if best_data is None or extracted_data['confidence'] > best_data['confidence']:
                        best_data = extracted_data
                return best_data if best_data is not None else {
                    'amount': None,
                    'date': None,
                    'bank_name': None,
                    'account_number': None,
                    'raw_text': '',
                    'confidence': 0.0
                }
            else:
                # Read original and produce many variants including halves
                original = cv2.imread(image_path)
                if original is None:
                    raise ValueError(f"Could not read image: {image_path}")
                candidates: List[np.ndarray] = []
                for sub in self._maybe_split_halves(original):
                    candidates.extend(self._generate_variants(sub))
                text = self._extract_best_from_variants(candidates)
                if mode == "ocr":
                    extracted_data = {
                        'amount': self.extract_amount(text),
                        'date': self.extract_date(text),
                        'bank_name': self.extract_bank_name(text),
                        'account_number': self.extract_account_number(text),
                        'raw_text': text,
                        'confidence': self.calculate_confidence({
                            'amount': self.extract_amount(text),
                            'date': self.extract_date(text),
                            'bank_name': self.extract_bank_name(text),
                            'account_number': self.extract_account_number(text)
                        }),
                        'processing_details': {
                            'mode_used': 'ocr',
                            'amount_source': 'ocr' if self.extract_amount(text) else None,
                            'date_source': 'ocr' if self.extract_date(text) else None,
                            'bank_source': 'ocr' if self.extract_bank_name(text) else None,
                            'account_source': 'ocr' if self.extract_account_number(text) else None,
                            'ocr_confidence': self.calculate_confidence({
                                'amount': self.extract_amount(text),
                                'date': self.extract_date(text),
                                'bank_name': self.extract_bank_name(text),
                                'account_number': self.extract_account_number(text)
                            })
                        }
                    }
                elif mode == "llm":
                    llm_data, llm_conf = await self.llm_parser.parse(text)
                    extracted_data = {
                        'amount': llm_data.get('amount'),
                        'date': llm_data.get('date'),
                        'bank_name': llm_data.get('bank_name'),
                        'account_number': llm_data.get('account_number'),
                        'raw_text': text,
                        'confidence': llm_conf
                    }
                else:
                    extracted_data = {
                        'amount': self.extract_amount(text),
                        'date': self.extract_date(text),
                        'bank_name': self.extract_bank_name(text),
                        'account_number': self.extract_account_number(text),
                        'raw_text': text,
                        'confidence': self.calculate_confidence({
                            'amount': self.extract_amount(text),
                            'date': self.extract_date(text),
                            'bank_name': self.extract_bank_name(text),
                            'account_number': self.extract_account_number(text)
                        })
                    }
                return extracted_data
        except Exception as e:
            return {
                'error': str(e),
                'amount': None,
                'date': None,
                'bank_name': None,
                'account_number': None,
                'raw_text': '',
                'confidence': 0.0
            }


