from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple


class LLMSlipParser:
    """LLM-backed parser for deposit slips.

    Uses OpenAI chat models to extract fields from OCR text. Falls back gracefully
    when API key is missing or an error occurs.
    """

    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
                    out["amount"] = float(amt_str)
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
        if not self._api_key:
            # No key: return empty
            print(f"🤖 LLM: No API key available, returning empty result")
            empty = {"amount": None, "date": None, "bank_name": None, "account_number": None}
            return empty, 0.0

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            system = (
                "You extract structured fields from deposit slip text."
                " Return strict JSON with keys: amount (number), date (YYYY-MM-DD),"
                " bank_name (string in {SBI,HDFC,ICICI,AXIS,PNB,DCB} if possible),"
                " account_number (digits only). If unknown, use null."
            )
            user = (
                "Extract fields from the following text."
                "\nTEXT:\n" + (text or "") +
                "\nJSON ONLY with keys [amount,date,bank_name,account_number]."
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
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


