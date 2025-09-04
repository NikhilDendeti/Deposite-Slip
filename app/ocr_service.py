import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import time
from typing import Dict, Optional, List
from pathlib import Path
import fitz  # PyMuPDF
from .slip_rule_parser import RuleBasedSlipParser
from .llm_parser import LLMSlipParser
from .gcv_client import GCVClient

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
            # Label-based with optional currency markers (₹, Rs, INR, $)
            r'(?:amount|total|deposit)[:\s]*[₹\$]?\s*(?:rs\.?|inr)?\s*([0-9,]+\.?[0-9]*)',
            # Standalone currency symbol before amount
            r'[₹\$]\s*([0-9,]+(?:\.[0-9]{1,2})?)',
            # Prefix currency code/word (Rs., INR)
            r'(?:rs\.?|inr)\s*([0-9,]+(?:\.[0-9]{1,2})?)',
        ]

        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}',
        ]

        self.bank_patterns = {
            'SBI': r'state\s+bank\s+of\s+india|\bsbi\b',
            'HDFC': r'hdfc\s+bank|\bhdfc\b',
            'ICICI': r'icici\s+bank|\bicici\b',
            'AXIS': r'axis\s+bank|\baxis\b',
            'PNB': r'punjab\s+national\s+bank|punjab\s+national|\bpnb\b',
            'DCB': r'dcb\s*bank|\bdcb\b',
            'BOB': r'bank\s+of\s+baroda|\bbob\b',
            'BOI': r'bank\s+of\s+india\b|\bboi\b',
            'CANARA': r'canara\s+bank',
            'KOTAK': r'kotak\s+mahindra\s+bank|\bkotak\b',
            'YES': r'yes\s+bank\b',
            'UNION': r'union\s+bank\s+of\s+india|union\s+bank',
            'IDBI': r'idbi\s+bank|\bidbi\b',
            'INDUSIND': r'indusind\s+bank|indus\s*ind',
            'FEDERAL': r'federal\s+bank\b',
            'RBL': r'rbl\s+bank\b',
            'BANDHAN': r'bandhan\s+bank\b',
            'AU': r'au\s+small\s+finance\s+bank|\bau\b',
            'UCO': r'uco\s+bank\b',
            'CENTRAL': r'central\s+bank\s+of\s+india|central\s+bank',
            'INDIANBANK': r'indian\s+bank\b',
            'IOB': r'indian\s+overseas\s+bank|\biob\b',
            'KVB': r'karur\s+vysya\s+bank|\bkvb\b',
            'SIB': r'south\s+indian\s+bank|\bsib\b',
            'IDFC': r'idfc\s+first\s+bank|\bidfc\b',
            'CSB': r'catholic\s+syrian\s+bank|\bcsb\b',
        }
        self.rule_parser = RuleBasedSlipParser()
        self.llm_parser = LLMSlipParser()
        self.paddle_ocr = None
        self.gcv_client = GCVClient()
        if _paddle_available:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("🧠 PaddleOCR initialized")
            except Exception:
                self.paddle_ocr = None

    # --- Image preprocessing helpers --------------------------------------------------------
    def _rotate_image(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        try:
            bgr = self._ensure_bgr(image)
            h, w = bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return image

    def _detect_and_rotate_upright(self, image: np.ndarray) -> np.ndarray:
        """Attempt to detect orientation with Tesseract OSD and rotate upright."""
        try:
            bgr = self._ensure_bgr(image)
            osd = pytesseract.image_to_osd(bgr, config='--psm 0 -l eng')
            # Typical OSD contains a line like: "Rotate: 90"
            match = re.search(r"Rotate:\s*(\d+)", osd)
            if match:
                rot = int(match.group(1)) % 360
                if rot != 0:
                    # Rotate in the opposite direction to make it upright
                    return self._rotate_image(bgr, -rot)
            return bgr
        except Exception:
            return image
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

        # Try to correct orientation first
        upright = self._detect_and_rotate_upright(bgr)

        # Add base rotations (0/90/180/270)
        base_rotations: List[np.ndarray] = [
            upright,
            cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(upright, cv2.ROTATE_180),
            cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        # Mild denoising to reduce background speckle
        denoised_rots: List[np.ndarray] = []
        for base in base_rotations:
            try:
                den = cv2.fastNlMeansDenoisingColored(self._ensure_bgr(base), None, 5, 5, 7, 21)
                denoised_rots.append(den)
            except Exception:
                denoised_rots.append(self._ensure_bgr(base))

        for base in denoised_rots:
            variants.append(base)

        # Grayscale and CLAHE
        gray = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY)
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
        up_bgr = cv2.resize(upright, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
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

    def render_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
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

    def _clean_text(self, text: str) -> str:
        """Remove obviously noisy lines and normalize spacing."""
        if not text:
            return ""
        cleaned_lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            alnum_count = len(re.findall(r"[A-Za-z0-9]", line))
            # Skip lines with too few alphanumerics or mostly punctuation
            if alnum_count < 3:
                continue
            non_alnum = len(re.findall(r"[^A-Za-z0-9]", line))
            if non_alnum > 0 and non_alnum / max(len(line), 1) > 0.6:
                continue
            # Skip lines that are mostly the same character repeated
            if re.search(r"^(.)\1{4,}$", re.sub(r"\s+", "", line)):
                continue
            # Compress internal spaces
            line = re.sub(r"\s+", " ", line)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def extract_text(self, image: np.ndarray) -> str:
        """Run OCR with several configs and optionally PaddleOCR, pick the longest output."""
        whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-/: ₹'
        configs = [
            f'-l eng --oem 1 --psm 6 -c tessedit_char_whitelist={whitelist} -c preserve_interword_spaces=1 -c user_defined_dpi=300',
            f'-l eng --oem 1 --psm 7 -c tessedit_char_whitelist={whitelist} -c preserve_interword_spaces=1 -c user_defined_dpi=300',
            f'-l eng --oem 1 --psm 11 -c tessedit_char_whitelist={whitelist} -c preserve_interword_spaces=1 -c user_defined_dpi=300',
            '-l eng --oem 3 --psm 4 -c preserve_interword_spaces=1 -c user_defined_dpi=300',
            '-l eng --oem 1 --psm 12 -c preserve_interword_spaces=1 -c user_defined_dpi=300'
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

        return self._clean_text(best_text)

    def _extract_best_from_variants(self, images: List[np.ndarray]) -> str:
        """Run OCR across multiple image variants and return the most informative text."""
        best_text = ''
        for img in images:
            text = self.extract_text(img)
            if len(text) > len(best_text):
                best_text = text
        return best_text

    def _extract_best_data_from_variants_ocr(self, images: List[np.ndarray]) -> Dict:
        """Run OCR across variants and aggregate with field-wise voting for robustness."""
        best_by_conf: Optional[Dict] = None
        amount_votes: Dict[str, int] = {}
        date_votes: Dict[str, int] = {}
        bank_votes: Dict[str, int] = {}
        acct_votes: Dict[str, int] = {}
        representative_text = ''

        for img in images:
            text = self.extract_text(img)
            amount = self.extract_amount(text)
            date_val = self.extract_date(text)
            bank = self.extract_bank_name(text)
            acct = self.extract_account_number(text)
            data = {
                'amount': amount,
                'date': date_val,
                'bank_name': bank,
                'account_number': acct,
                'raw_text': text,
            }
            conf = self.calculate_confidence(data)
            data['confidence'] = conf
            if best_by_conf is None or conf > best_by_conf.get('confidence', 0.0):
                best_by_conf = data
                representative_text = text
            if amount is not None:
                key = f"{amount:.2f}"
                amount_votes[key] = amount_votes.get(key, 0) + 1
            if date_val:
                date_votes[date_val] = date_votes.get(date_val, 0) + 1
            if bank:
                bank_votes[bank] = bank_votes.get(bank, 0) + 1
            if acct:
                acct_votes[acct] = acct_votes.get(acct, 0) + 1

        def pick_majority(votes: Dict[str, int]) -> Optional[str]:
            if not votes:
                return None
            return max(votes.items(), key=lambda kv: kv[1])[0]

        majority_amount_key = pick_majority(amount_votes)
        majority_amount = float(majority_amount_key) if majority_amount_key else None
        majority_date = pick_majority(date_votes)
        majority_bank = pick_majority(bank_votes)
        majority_acct = pick_majority(acct_votes)

        aggregated = {
            'amount': majority_amount if majority_amount is not None else (best_by_conf.get('amount') if best_by_conf else None),
            'date': majority_date if majority_date else (best_by_conf.get('date') if best_by_conf else None),
            'bank_name': majority_bank if majority_bank else (best_by_conf.get('bank_name') if best_by_conf else None),
            'account_number': majority_acct if majority_acct else (best_by_conf.get('account_number') if best_by_conf else None),
            'raw_text': representative_text if representative_text else (best_by_conf.get('raw_text') if best_by_conf else ''),
        }
        aggregated['confidence'] = self.calculate_confidence(aggregated)
        return aggregated

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

        denom_cues = [
            r"\bdenomination\b", r"\bcoins?\b", r"\bnotes?\b",
            r"\b(2000|1000|500|200|100|50|20|10|5|2|1)\s*[xX*]\s*\d+\b",
        ]

        def is_denom_context(span_start: int, span_end: int, haystack: str) -> bool:
            window = haystack[max(0, span_start - 60): min(len(haystack), span_end + 60)]
            for cue in denom_cues:
                if re.search(cue, window, re.IGNORECASE):
                    return True
            return False

        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 0.01 <= amount <= 10000000 and not is_denom_context(match.start(), match.end(), text_lower):
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
                    if 0.01 <= amount <= 10000000 and not is_denom_context(match.start(), match.end(), normalized):
                        return amount
                except (ValueError, IndexError):
                    continue
        return None

    def extract_date(self, text: str) -> Optional[str]:
        if not text:
            return None
        # First try label-based forms with OCR confusions
        def normalize_date_chars(s: str) -> str:
            table = str.maketrans({
                'O': '0', 'o': '0',
                'l': '1', 'I': '1', '|': '1',
                'S': '5', 's': '5',
                'B': '8', 'b': '8',
            })
            return s.translate(table)

        text_norm = normalize_date_chars(text)

        label_patterns = [
            r'(?:date|dated|dt)\s*[:\-]?\s*([0-9OolS]{1,2}[./\-][0-9OolS]{1,2}[./\-][0-9OolS]{2,4})',
            r'(?:date|dated|dt)\s*[:\-]?\s*([0-9]{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*[0-9]{2,4})',
        ]
        for pat in label_patterns:
            m = re.search(pat, text_norm, re.IGNORECASE)
            if m:
                return self.standardize_date(m.group(1))

        # Then try general patterns on normalized text
        extra_patterns = [
            r'(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})',
            r'(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*\d{2,4})',
        ] + self.date_patterns
        for pattern in extra_patterns:
            matches = re.search(pattern, text_norm, re.IGNORECASE)
            if matches:
                date_str = matches.group(1)
                return self.standardize_date(date_str)
        return None

    def standardize_date(self, date_str: str) -> Optional[str]:
        from datetime import datetime
        import difflib
        # First: straightforward formats (Indian first)
        formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%Y-%m-%d', '%Y/%m/%d',
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
        ]
        s = (date_str or '').strip()
        for fmt in formats:
            try:
                date_obj = datetime.strptime(s, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        # Second: textual months with separators
        text_formats = [
            '%d %b %Y', '%d %b %y', '%d %B %Y', '%d %B %y',
            '%d-%b-%Y', '%d-%b-%y', '%d-%B-%Y', '%d-%B-%y',
            '%d.%b.%Y', '%d.%b.%y', '%d.%B.%Y', '%d.%B.%y',
        ]
        s_norm_spaces = re.sub(r"\s+", " ", s)
        for fmt in text_formats:
            try:
                date_obj = datetime.strptime(s_norm_spaces, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        # Third: robust cleanup for OCR-noisy tokens like "2JANDDE51"
        if s:
            # Normalize confusable glyphs
            trans = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1', '|': '1', 'S': '5', 'B': '8'})
            s2 = s.translate(trans)
            # Insert separators between digits and letters
            s2 = re.sub(r'(?<=\d)(?=[A-Za-z])', '-', s2)
            s2 = re.sub(r'(?<=[A-Za-z])(?=\d)', '-', s2)
            # Collapse repeated letters
            s2 = re.sub(r'([A-Za-z])\1{1,}', r'\1', s2)
            # Keep only alnum and dashes/spaces
            s2 = re.sub(r'[^A-Za-z0-9\-\s]', '', s2)
            tokens = [t for t in re.split(r'[\s\-]+', s2) if t]
            if len(tokens) >= 2:
                # Expect day, month(word/num), year(optional)
                day_token = tokens[0]
                month_token = tokens[1]
                year_token = tokens[2] if len(tokens) >= 3 else ''
                # Fuzzy month mapping
                month_map = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                }
                mtok = re.sub(r'[^A-Za-z]', '', month_token).lower()
                mtok = re.sub(r'(.)\1+', r'\1', mtok)
                month_num: Optional[int] = None
                if mtok.isdigit():
                    try:
                        month_num = int(mtok)
                    except Exception:
                        month_num = None
                else:
                    choices = list(month_map.keys())
                    close = difflib.get_close_matches(mtok, choices, n=1, cutoff=0.5)
                    if close:
                        month_num = month_map[close[0]]
                # Parse day
                try:
                    day_num = int(re.sub(r'\D', '', day_token))
                except Exception:
                    day_num = None
                # Parse year (assume 20YY for 2-digit)
                ydigits = re.sub(r'\D', '', year_token)
                year_num: Optional[int] = None
                if ydigits:
                    try:
                        y = int(ydigits)
                        if y < 100:
                            year_num = 2000 + y
                        else:
                            year_num = y
                    except Exception:
                        year_num = None
                # If year missing but there is a plausible 4-digit elsewhere, try to pick it
                if year_num is None and len(tokens) >= 4:
                    y2 = re.sub(r'\D', '', tokens[3])
                    if len(y2) in (2, 4):
                        try:
                            y = int(y2)
                            year_num = 2000 + y if y < 100 else y
                        except Exception:
                            pass
                if day_num and month_num and 1 <= day_num <= 31 and 1 <= month_num <= 12 and year_num:
                    try:
                        dt = datetime(year_num, month_num, day_num)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        pass
        # Give up
        return None

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
        # LLM mode disabled: fallback to OCR to keep flow working
        if (mode or "").lower() == "llm":
            print("⚠️ LLM mode is disabled; falling back to OCR mode")
            mode = "ocr"
        try:
            total_start = time.perf_counter()
            lower_path = image_path.lower()
            if lower_path.endswith('.pdf'):
                print(f"📄 PDF DETECTED: Processing PDF file")
                page_images = self.render_pdf_to_images(image_path)
                if not page_images:
                    raise ValueError("Could not render any pages from PDF")
                print(f"📄 PDF RENDERED: {len(page_images)} pages")

                best_data: Optional[Dict] = None
                best_conf = -1.0
                for i, page_image in enumerate(page_images):
                    print(f"📄 PROCESSING PAGE {i+1}/{len(page_images)}")
                    # Try page, its halves, and multiple preprocess variants
                    ocr_text_start = time.perf_counter()
                    candidates: List[np.ndarray] = []
                    for sub in self._maybe_split_halves(page_image):
                        candidates.extend(self._generate_variants(sub))
                    # Prefer text from the best-scoring variant by field-confidence
                    best_data_for_ocr = self._extract_best_data_from_variants_ocr(candidates)
                    text = best_data_for_ocr.get('raw_text', '') or self._extract_best_from_variants(candidates)
                    ocr_text_ms = int((time.perf_counter() - ocr_text_start) * 1000)
                    # Print full OCR text for inspection
                    try:
                        print(f"📝 FULL OCR TEXT (page {i+1}):\n{text}\n--- END PAGE {i+1} ---")
                        print("__________________________________________________________________")
                        print(f"⏱ OCR(text) time (page {i+1}): {ocr_text_ms} ms")
                    except Exception:
                        pass
                    if mode == "ocr":
                        print(f"🔍 USING OCR MODE")
                        best_variant_data = best_data_for_ocr
                        extracted_data = {
                            'amount': best_variant_data.get('amount'),
                            'date': best_variant_data.get('date'),
                            'bank_name': best_variant_data.get('bank_name'),
                            'account_number': best_variant_data.get('account_number'),
                            'raw_text': best_variant_data.get('raw_text', text),
                            'confidence': best_variant_data.get('confidence', 0.0),
                            'processing_details': {
                                'mode_used': 'ocr',
                                'amount_source': 'ocr' if best_variant_data.get('amount') else None,
                                'date_source': 'ocr' if best_variant_data.get('date') else None,
                                'bank_source': 'ocr' if best_variant_data.get('bank_name') else None,
                                'account_source': 'ocr' if best_variant_data.get('account_number') else None,
                                'ocr_confidence': best_variant_data.get('confidence', 0.0),
                                'timings': {
                                    'ocr_text_ms': ocr_text_ms
                                }
                            }
                        }
                    elif mode == "llm":
                        print(f"🤖 USING LLM MODE")
                        llm_start = time.perf_counter()
                        llm_data, llm_conf = await self.llm_parser.parse(text)
                        llm_ms = int((time.perf_counter() - llm_start) * 1000)
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
                                'llm_confidence': llm_conf,
                                'timings': {
                                    'ocr_text_ms': ocr_text_ms,
                                    'llm_ms': llm_ms
                                }
                            }
                        }
                    elif mode == "vision":
                        print(f"🖼️ USING VISION MODE")
                        # Use original page image for vision model
                        vision_start = time.perf_counter()
                        # Provide multiple variants and OCR text as hint to vision model
                        images_for_vision: List[np.ndarray] = [page_image]
                        for sub in self._maybe_split_halves(page_image):
                            images_for_vision.extend(self._generate_variants(sub)[:2])
                        vision_data, vision_conf = await self.llm_parser.parse_images_cv(images_for_vision[:4], text)
                        vision_ms = int((time.perf_counter() - vision_start) * 1000)
                        print(f"🖼️ VISION RESULT: {vision_data}, confidence: {vision_conf}")
                        extracted_data = {
                            'amount': vision_data.get('amount'),
                            'date': vision_data.get('date'),
                            'bank_name': vision_data.get('bank_name'),
                            'account_number': vision_data.get('account_number'),
                            'raw_text': text,
                            'confidence': vision_conf,
                            'processing_details': {
                                'mode_used': 'vision',
                                'amount_source': 'vision' if vision_data.get('amount') is not None else None,
                                'date_source': 'vision' if vision_data.get('date') else None,
                                'bank_source': 'vision' if vision_data.get('bank_name') else None,
                                'account_source': 'vision' if vision_data.get('account_number') else None,
                                'llm_confidence': vision_conf,
                                'timings': {
                                    'ocr_text_ms': ocr_text_ms,
                                    'vision_ms': vision_ms
                                }
                            }
                        }
                    elif mode == "gcv":
                        print("🧭 USING GOOGLE CLOUD VISION (GCV) MODE")
                        gcv_start = time.perf_counter()
                        # Prefer the original rendered page image for GCV
                        text_gcv, conf_gcv = await self.gcv_client.detect_text_from_cv2(page_image, feature_type="DOCUMENT_TEXT_DETECTION", language_hints=["en"])
                        gcv_ms = int((time.perf_counter() - gcv_start) * 1000)
                        gcv_best = {
                            'amount': self.extract_amount(text_gcv),
                            'date': self.extract_date(text_gcv),
                            'bank_name': self.extract_bank_name(text_gcv),
                            'account_number': self.extract_account_number(text_gcv),
                            'raw_text': self._clean_text(text_gcv),
                        }
                        # Optional OpenAI post-processing
                        openai_amount = None
                        openai_date = None
                        openai_bank = None
                        openai_acct = None
                        try:
                            parsed, _raw = await self.gcv_client.post_process_with_openai(text_gcv)
                            if parsed:
                                try:
                                    a = parsed.get('amount')
                                    if a is not None:
                                        openai_amount = float(a)
                                except Exception:
                                    openai_amount = None
                                d = parsed.get('date')
                                if d:
                                    openai_date = str(d)
                                openai_bank = parsed.get('bank_name') or None
                                openai_acct = parsed.get('account_number') or None
                        except Exception:
                            pass
                        merged_amount = openai_amount if openai_amount is not None else gcv_best.get('amount')
                        merged_date = openai_date if openai_date else gcv_best.get('date')
                        merged_bank = openai_bank if openai_bank else gcv_best.get('bank_name')
                        merged_acct = openai_acct if openai_acct else gcv_best.get('account_number')
                        merged = {
                            'amount': merged_amount,
                            'date': merged_date,
                            'bank_name': merged_bank,
                            'account_number': merged_acct,
                            'raw_text': gcv_best.get('raw_text', ''),
                        }
                        merged['confidence'] = max(self.calculate_confidence(merged), float(conf_gcv) if conf_gcv is not None else 0.0)
                        extracted_data = {
                            'amount': merged.get('amount'),
                            'date': merged.get('date'),
                            'bank_name': merged.get('bank_name'),
                            'account_number': merged.get('account_number'),
                            'raw_text': merged.get('raw_text', ''),
                            'confidence': merged.get('confidence', 0.0),
                            'processing_details': {
                                'mode_used': 'gcv',
                                'amount_source': 'gcv_openai' if openai_amount is not None else ('gcv' if gcv_best.get('amount') else None),
                                'date_source': 'gcv_openai' if openai_date else ('gcv' if gcv_best.get('date') else None),
                                'bank_source': 'gcv_openai' if openai_bank else ('gcv' if gcv_best.get('bank_name') else None),
                                'account_source': 'gcv_openai' if openai_acct else ('gcv' if gcv_best.get('account_number') else None),
                                'gcv_confidence': conf_gcv,
                                'timings': {
                                    'ocr_text_ms': ocr_text_ms,
                                    'gcv_ms': gcv_ms
                                }
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
                    if extracted_data['confidence'] > best_conf:
                        best_conf = extracted_data['confidence']
                        best_data = extracted_data
                total_ms = int((time.perf_counter() - total_start) * 1000)
                if best_data is not None:
                    best_data.setdefault('processing_details', {}).setdefault('timings', {})['total_ms'] = total_ms
                    print(f"⏱ Total processing time: {total_ms} ms")
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
                ocr_text_start = time.perf_counter()
                candidates: List[np.ndarray] = []
                for sub in self._maybe_split_halves(original):
                    candidates.extend(self._generate_variants(sub))
                # Prefer text from the best-scoring variant by field-confidence
                best_data_for_ocr = self._extract_best_data_from_variants_ocr(candidates)
                text = best_data_for_ocr.get('raw_text', '') or self._extract_best_from_variants(candidates)
                ocr_text_ms = int((time.perf_counter() - ocr_text_start) * 1000)
                # Print full OCR text for inspection
                try:
                    print(f"📝 FULL OCR TEXT:\n{text}\n--- END OCR TEXT ---")
                    print("__________________________________________________________________")
                    print(f"⏱ OCR(text) time: {ocr_text_ms} ms")
                except Exception:
                    pass
                if mode == "ocr":
                    best_variant_data = best_data_for_ocr
                    extracted_data = {
                        'amount': best_variant_data.get('amount'),
                        'date': best_variant_data.get('date'),
                        'bank_name': best_variant_data.get('bank_name'),
                        'account_number': best_variant_data.get('account_number'),
                        'raw_text': best_variant_data.get('raw_text', text),
                        'confidence': best_variant_data.get('confidence', 0.0),
                        'processing_details': {
                            'mode_used': 'ocr',
                            'amount_source': 'ocr' if best_variant_data.get('amount') else None,
                            'date_source': 'ocr' if best_variant_data.get('date') else None,
                            'bank_source': 'ocr' if best_variant_data.get('bank_name') else None,
                            'account_source': 'ocr' if best_variant_data.get('account_number') else None,
                            'ocr_confidence': best_variant_data.get('confidence', 0.0),
                            'timings': {
                                'ocr_text_ms': ocr_text_ms
                            }
                        }
                    }
                elif mode == "llm":
                    llm_start = time.perf_counter()
                    llm_data, llm_conf = await self.llm_parser.parse(text)
                    llm_ms = int((time.perf_counter() - llm_start) * 1000)
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
                            'llm_confidence': llm_conf,
                            'timings': {
                                'ocr_text_ms': ocr_text_ms,
                                'llm_ms': llm_ms
                            }
                        }
                    }
                elif mode == "vision":
                    vision_start = time.perf_counter()
                    # Provide multiple variants and OCR text as hint to vision model
                    images_for_vision: List[np.ndarray] = [original]
                    for sub in self._maybe_split_halves(original):
                        images_for_vision.extend(self._generate_variants(sub)[:2])
                    vision_data, vision_conf = await self.llm_parser.parse_images_cv(images_for_vision[:4], text)
                    vision_ms = int((time.perf_counter() - vision_start) * 1000)
                    extracted_data = {
                        'amount': vision_data.get('amount'),
                        'date': vision_data.get('date'),
                        'bank_name': vision_data.get('bank_name'),
                        'account_number': vision_data.get('account_number'),
                        'raw_text': text,
                        'confidence': vision_conf,
                        'processing_details': {
                            'mode_used': 'vision',
                            'amount_source': 'vision' if vision_data.get('amount') is not None else None,
                            'date_source': 'vision' if vision_data.get('date') else None,
                            'bank_source': 'vision' if vision_data.get('bank_name') else None,
                            'account_source': 'vision' if vision_data.get('account_number') else None,
                            'llm_confidence': vision_conf,
                            'timings': {
                                'ocr_text_ms': ocr_text_ms,
                                'vision_ms': vision_ms
                            }
                        }
                    }
                elif mode == "gcv":
                    gcv_start = time.perf_counter()
                    # Use original image for GCV
                    text_gcv, conf_gcv = await self.gcv_client.detect_text_from_cv2(original, feature_type="DOCUMENT_TEXT_DETECTION", language_hints=["en"])
                    gcv_ms = int((time.perf_counter() - gcv_start) * 1000)
                    gcv_best = {
                        'amount': self.extract_amount(text_gcv),
                        'date': self.extract_date(text_gcv),
                        'bank_name': self.extract_bank_name(text_gcv),
                        'account_number': self.extract_account_number(text_gcv),
                        'raw_text': self._clean_text(text_gcv),
                    }
                    # Optional OpenAI post-processing
                    openai_amount = None
                    openai_date = None
                    openai_bank = None
                    openai_acct = None
                    try:
                        parsed, _raw = await self.gcv_client.post_process_with_openai(text_gcv)
                        if parsed:
                            try:
                                a = parsed.get('amount')
                                if a is not None:
                                    openai_amount = float(a)
                            except Exception:
                                openai_amount = None
                            d = parsed.get('date')
                            if d:
                                openai_date = str(d)
                            openai_bank = parsed.get('bank_name') or None
                            openai_acct = parsed.get('account_number') or None
                    except Exception:
                        pass
                    merged_amount = openai_amount if openai_amount is not None else gcv_best.get('amount')
                    merged_date = openai_date if openai_date else gcv_best.get('date')
                    merged_bank = openai_bank if openai_bank else gcv_best.get('bank_name')
                    merged_acct = openai_acct if openai_acct else gcv_best.get('account_number')
                    merged = {
                        'amount': merged_amount,
                        'date': merged_date,
                        'bank_name': merged_bank,
                        'account_number': merged_acct,
                        'raw_text': gcv_best.get('raw_text', ''),
                    }
                    merged['confidence'] = max(self.calculate_confidence(merged), float(conf_gcv) if conf_gcv is not None else 0.0)
                    extracted_data = {
                        'amount': merged.get('amount'),
                        'date': merged.get('date'),
                        'bank_name': merged.get('bank_name'),
                        'account_number': merged.get('account_number'),
                        'raw_text': gcv_best.get('raw_text', ''),
                        'confidence': merged.get('confidence', 0.0),
                        'processing_details': {
                            'mode_used': 'gcv',
                            'amount_source': 'gcv_openai' if openai_amount is not None else ('gcv' if gcv_best.get('amount') else None),
                            'date_source': 'gcv_openai' if openai_date else ('gcv' if gcv_best.get('date') else None),
                            'bank_source': 'gcv_openai' if openai_bank else ('gcv' if gcv_best.get('bank_name') else None),
                            'account_source': 'gcv_openai' if openai_acct else ('gcv' if gcv_best.get('account_number') else None),
                            'gcv_confidence': conf_gcv,
                            'timings': {
                                'ocr_text_ms': ocr_text_ms,
                                'gcv_ms': gcv_ms
                            }
                        }
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
                total_ms = int((time.perf_counter() - total_start) * 1000)
                extracted_data.setdefault('processing_details', {}).setdefault('timings', {})['total_ms'] = total_ms
                print(f"⏱ Total processing time: {total_ms} ms")
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


