from __future__ import annotations

import json
import os
import re
import base64
import mimetypes
from typing import Dict, Optional, Tuple, List

import cv2  # type: ignore
import numpy as np  # type: ignore


class LLMSlipParser:
    """LLM-backed parser for deposit slips.

    Uses OpenAI chat models to extract fields from OCR text. Falls back gracefully
    when API key is missing or an error occurs.
    """

    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Separate override for a vision-capable model (chat multimodal)
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self._api_key = os.getenv("OPENAI_API_KEY")

    def _score(self, data: Dict[str, Optional[str]]) -> float:
        score = 0.0
        if data.get("amount") not in (None, ""):
            score += 0.45
        if data.get("date"):
            score += 0.25
        if data.get("bank_name"):
            score += 0.1
        if data.get("account_number"):
            score += 0.15
        if sum(1 for k in ("amount", "date", "bank_name", "account_number") if data.get(k)) >= 2:
            score += 0.05
        return min(score, 1.0)

    def _coerce(self, raw: Dict) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {
            "amount": None,
            "date": None,
            "bank_name": None,
            "account_number": None,
        }
        try:
            if raw.get("amount") is not None:
                amt_str = str(raw.get("amount")).replace(",", "").strip()
                if re.match(r"^-?\d+(?:\.\d+)?$", amt_str):
                    val = float(amt_str)
                    # Enforce plausible INR range and non-negative with up to 2 decimals
                    if 0.01 <= val <= 10000000:
                        out["amount"] = round(val, 2)
            if raw.get("date"):
                out["date"] = str(raw.get("date")).strip()
            if raw.get("bank_name"):
                out["bank_name"] = str(raw.get("bank_name")).strip()
            if raw.get("account_number"):
                digits = re.sub(r"\D", "", str(raw.get("account_number")))
                if 6 <= len(digits) <= 20:
                    out["account_number"] = digits
        except Exception:
            pass
        return out

    async def parse(self, text: str) -> Tuple[Dict[str, Optional[str]], float]:
        print(f"🤖 LLMParser.parse: Starting LLM-based parsing")
        # Log full input text passed to LLM
        try:
            preview_len = len(text or "")
            print(f"🤖 LLM INPUT TEXT ({preview_len} chars):\n{text}\n--- END LLM INPUT ---")
        except Exception:
            pass
        if not self._api_key:
            # No key: return empty
            print(f"🤖 LLM: No API key available, returning empty result")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            # Optional rule-based hints to guide the LLM conservatively
            hints: Dict[str, Optional[str]] = {}
            try:
                from .slip_rule_parser import RuleBasedSlipParser  # type: ignore
                _rb = RuleBasedSlipParser()
                rb_result, _rb_conf = _rb.parse(text or "")
                hints = rb_result or {}
            except Exception:
                hints = {}
            system = (
                "You are an expert data extractor for INDIAN bank deposit slips. "
                "Your task is to return STRICT JSON with keys exactly: "
                "amount (number), date (YYYY-MM-DD), bank_name (string), account_number (digits only). "
                "If any field is not confidently found, output null for that field. Do not add extra keys.\n\n"

                "Guidelines:"\
                "\n- Consider only deposit slip content. Ignore headers/footers unrelated to the slip. "
                "\n- Prefer values adjacent to labels: Amount/Total/Deposit, Date/Dt/Dated, A/C No/Account No/Account Number. "
                "\n- If multiple candidates exist, choose the one closest to the label. "
                "\n- Treat OCR noise as junk: ignore lines that are mostly punctuation, symbols, or repeated characters. "
                "\n- Amount: in INR rupees. Prefer numeric over words when both are present. Strip commas and currency symbols (₹, Rs, INR). "
                " Reject amounts with more than 2 decimals or implausible magnitudes (<=0 or >10,000,000). "
                " Ignore denomination tables like '2000X', '500X', '100X', and their totals; do not sum denominations. "
                "\n- Date: normalize to YYYY-MM-DD. Prefer Indian formats (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY). "
                " If day and month are ambiguous, assume DD/MM/YYYY. Accept textual months (e.g., 04 Sep 2025). "
                " Reject invalid dates (day>31, month>12). For 2-digit years, assume 20YY. "
                "\n- Bank name: map to whitelist when possible via fuzzy/partial matches: "
                "[SBI,HDFC,ICICI,AXIS,PNB,DCB,BOB,BOI,CANARA,KOTAK,YES,UNION,IDBI,INDUSIND,FEDERAL,RBL,BANDHAN,AU,UCO,CENTRAL,INDIANBANK,IOB,KVB,SIB,IDFC,CSB]. "
                " Examples: 'State Bank of India' or 'S.B.I.' -> SBI. If no clear mapping, use null. "
                "\n- Account number: output digits only (strip spaces/dashes). Length must be 6–20 digits; otherwise null. "
                " Prefer values near labels 'A/C No', 'A c No', 'Account No', 'Account Number'. "
                "\n- Be conservative: if uncertain, use null rather than guessing."
            )
            user = (
                "Extract the four fields from the following deposit slip text and return JSON ONLY with keys "
                "[amount,date,bank_name,account_number]. No prose.\n" \
                "TEXT:\n" + (text or "")
            )
            system_with_hints = system
            if any(v for v in hints.values()):
                try:
                    system_with_hints += (
                        "\n\nUse the following optional, rule-based HINTS only if corroborated by text context: "
                        + json.dumps({k: v for k, v in hints.items() if v})
                    )
                except Exception:
                    pass

            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_with_hints},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            cleaned = self._coerce(parsed)
            conf = self._score(cleaned)
            print(f"🤖 LLM: Extraction successful - {cleaned}, confidence: {conf}")
            return cleaned, conf
        except Exception as e:
            print(f"🤖 LLM: Extraction failed - {str(e)}")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0

    # -------------------- VISION (Chat Completions with image_url) --------------------
    def _image_path_to_data_url(self, image_path: str) -> Optional[str]:
        try:
            mime, _ = mimetypes.guess_type(image_path)
            if mime is None:
                # Default to jpeg if unknown
                mime = "image/jpeg"
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def _cv_image_to_data_url(self, image: np.ndarray) -> Optional[str]:
        try:
            # Encode to JPEG for compactness
            ok, buf = cv2.imencode('.jpg', image)
            if not ok:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    async def parse_image_data_url(self, data_url: str) -> Tuple[Dict[str, Optional[str]], float]:
        print(f"🤖 LLMParser.parse_image_data_url: Starting vision-based parsing")
        # Log data URL characteristics to understand input size/type
        try:
            prefix_preview = data_url[:64] if data_url else ""
            print(f"🖼️ VISION INPUT DATA URL: length={len(data_url or '')}, prefix={prefix_preview}...")
        except Exception:
            pass
        if not self._api_key:
            print(f"🤖 LLM: No API key available, returning empty result")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)

            system_instruction = (
                "You are a vision model that reads INDIAN bank deposit slip IMAGES and extracts fields. "
                "Return STRICT JSON with keys exactly: amount (number), date (YYYY-MM-DD), bank_name (string), account_number (digits only). "
                "If unknown, use null. Do not include any extra keys.\n\n"

                "Visual heuristics:"\
                "\n- Locate printed labels: Amount/Total/Deposit, Date/Dt/Dated, A/C No/Account No/Account Number. "
                " Extract the nearest clean value next to each label. "
                "\n- Ignore denomination tables (e.g., 2000X, 500X, 100X, 50X, Coins) and their totals—they are not the slip amount. "
                "\n- Amount: INR. Prefer numeric over words; strip commas and currency symbols; reject >2 decimals or >10,000,000. "
                "\n- Date: normalize to YYYY-MM-DD. Prefer Indian formats (DD/MM/YYYY). If ambiguous, assume DD/MM/YYYY. Accept textual months. Reject invalid dates. "
                "\n- Bank name: map to whitelist via fuzzy/partial matches when possible: "
                "{SBI,HDFC,ICICI,AXIS,PNB,DCB,BOB,BOI,CANARA,KOTAK,YES,UNION,IDBI,INDUSIND,FEDERAL,RBL,BANDHAN,AU,UCO,CENTRAL,INDIANBANK,IOB,KVB,SIB,IDFC,CSB}; else null. "
                "\n- Account number: digits only, 6–20 in length; prefer near labels. "
                "\n- Be conservative: return null if uncertain."
            )

            user_instruction = (
                "Extract the four fields from this deposit slip IMAGE and return JSON ONLY with keys "
                "[amount,date,bank_name,account_number]. No prose."
            )

            model_to_use = self.vision_model or self.model or "gpt-4o-mini"

            try:
                resp = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_instruction},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "{}") if resp and resp.choices else "{}"
            except Exception as inner_e:
                if model_to_use != "gpt-4o-mini":
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_instruction},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            },
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    content = (resp.choices[0].message.content or "{}") if resp and resp.choices else "{}"
                else:
                    raise inner_e

            parsed = json.loads(content)
            cleaned = self._coerce(parsed)
            conf = self._score(cleaned)
            print(f"🤖 LLM Vision: Extraction successful - {cleaned}, confidence: {conf}")
            return cleaned, conf
        except Exception as e:
            print(f"🤖 LLM Vision: Extraction failed - {str(e)}")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0

    async def parse_image_path(self, image_path: str) -> Tuple[Dict[str, Optional[str]], float]:
        data_url = self._image_path_to_data_url(image_path)
        if not data_url:
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0
        return await self.parse_image_data_url(data_url)

    async def parse_image_cv(self, image: np.ndarray) -> Tuple[Dict[str, Optional[str]], float]:
        # Backwards-compatible single-image entrypoint
        return await self.parse_images_cv([image], None)

    async def parse_images_cv(self, images: List[np.ndarray], text_hint: Optional[str]) -> Tuple[Dict[str, Optional[str]], float]:
        try:
            meta_list: List[str] = []
            for img in images or []:
                try:
                    h, w = img.shape[:2]
                    ch = img.shape[2] if hasattr(img, 'shape') and len(img.shape) == 3 else 1
                    meta_list.append(f"{h}x{w}x{ch}")
                except Exception:
                    meta_list.append("?")
            print(f"🖼️ VISION MULTI INPUT: count={len(images or [])}, metas={meta_list}, hint_len={len(text_hint or '')}")
        except Exception:
            pass

        if not self._api_key:
            print(f"🤖 LLM: No API key available, returning empty result")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)

            system_instruction = (
                "You are a vision model that reads INDIAN bank deposit slip IMAGES and extracts fields. "
                "Return STRICT JSON with keys exactly: amount (number), date (YYYY-MM-DD), bank_name (string), account_number (digits only). "
                "If unknown, use null. Do not include any extra keys.\n\n"

                "Visual heuristics:"\
                "\n- Locate printed labels: Amount/Total/Deposit, Date/Dt/Dated, A/C No/Account No/Account Number. "
                " Extract the nearest clean value next to each label. "
                "\n- Ignore denomination tables (e.g., 2000X, 500X, 100X, 50X, Coins) and their totals—they are not the slip amount. "
                "\n- Amount: INR. Prefer numeric over words; strip commas and currency symbols; reject >2 decimals or >10,000,000. "
                "\n- Date: normalize to YYYY-MM-DD. Prefer Indian formats (DD/MM/YYYY). If ambiguous, assume DD/MM/YYYY. Accept textual months. Reject invalid dates. "
                "\n- Bank name: map to whitelist via fuzzy/partial matches when possible: "
                "{SBI,HDFC,ICICI,AXIS,PNB,DCB,BOB,BOI,CANARA,KOTAK,YES,UNION,IDBI,INDUSIND,FEDERAL,RBL,BANDHAN,AU,UCO,CENTRAL,INDIANBANK,IOB,KVB,SIB,IDFC,CSB}; else null. "
                "\n- Account number: digits only, 6–20 in length; prefer near labels. "
                "\n- Be conservative: return null if uncertain."
            )

            user_blocks: List[Dict] = []
            user_blocks.append({"type": "text", "text": (
                "Extract the four fields from these deposit slip IMAGES and return JSON ONLY with keys "
                "[amount,date,bank_name,account_number]. No prose."
            )})
            if text_hint:
                # Provide OCR hint to assist reading small text; model must still verify visually
                user_blocks.append({"type": "text", "text": f"OCR hint text (may contain errors):\n{text_hint[:6000]}"})

            # Limit number of images to avoid excessive payload; prioritize first N
            max_imgs = 4
            for img in (images or [])[:max_imgs]:
                data_url = self._cv_image_to_data_url(img)
                if not data_url:
                    continue
                user_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

            model_to_use = self.vision_model or self.model or "gpt-4o-mini"

            try:
                resp = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_blocks},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = (resp.choices[0].message.content or "{}") if resp and resp.choices else "{}"
            except Exception as inner_e:
                if model_to_use != "gpt-4o-mini":
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_blocks},
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    content = (resp.choices[0].message.content or "{}") if resp and resp.choices else "{}"
                else:
                    raise inner_e

            parsed = json.loads(content)
            cleaned = self._coerce(parsed)
            conf = self._score(cleaned)
            print(f"🤖 LLM Vision(multi): Extraction successful - {cleaned}, confidence: {conf}")
            return cleaned, conf
        except Exception as e:
            print(f"🤖 LLM Vision(multi): Extraction failed - {str(e)}")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0


