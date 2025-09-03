"""Rule-based parser for Indian bank deposit slips.

The parser works on plain OCR text and uses weighted regex heuristics
to extract: amount, date, bank_name, and account_number. It is designed
to be robust against common slip templates (SBI, HDFC, ICICI, AXIS, PNB, DCB, etc.).

Accuracy goals: 90%+ on clear scans. Confidence reflects label proximity and
field presence.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, List
from datetime import datetime


class RuleBasedSlipParser:
    def __init__(self) -> None:
        # Common numeric patterns
        # 1) 12,34,567.89 or 1234567.89 or 12,34,567
        self.amount_number = r"(?:\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+\.\d{1,2}|\d{2,})"

        # Labels frequently appearing near amounts
        self.amount_labels = [
            r"total\s*amount", r"total\s*", r"cash\s*total", r"amount\s*", r"deposit\s*amount",
            r"sum\s*total", r"grand\s*total", r"total\s*\(in\s*words\)", r"in\s*words"
        ]

        # Date formats
        self.date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b",
            r"\b(?:date\s*[:\-]?\s*)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        ]

        # Bank name indicators (map to canonical)
        self.bank_map: Dict[str, str] = {
            r"\bdcb\s*bank\b": "DCB",
            r"\bstate\s*bank\s*of\s*india\b|\bsbi\b": "SBI",
            r"\bhdfc\s*bank\b|\bhdfc\b": "HDFC",
            r"\bicici\s*bank\b|\bicici\b": "ICICI",
            r"\baxis\s*bank\b|\baxis\b": "AXIS",
            r"\bpunjab\s*national\s*bank\b|\bpnb\b": "PNB",
        }

        # Account number patterns
        self.account_patterns = [
            r"a/?c\s*(?:no\.?|number)?\s*[:\-]?\s*(\d{6,20})",
            r"account\s*(?:no\.?|number)?\s*[:\-]?\s*(\d{6,20})",
            r"acc\s*no\.?\s*[:\-]?\s*(\d{6,20})",
        ]

        # Currency markers
        self.currency_markers = [r"\bINR\b", r"\bRs\.?\b", r"₹", r"\bRupees?\b"]

        # Amount in words basic vocabulary (English)
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
            'hundred': 100, 'thousand': 1000, 'lakh': 100000, 'lac': 100000,
            'lacs': 100000, 'lakhs': 100000, 'million': 1000000
        }

    # Public API
    def parse(self, text: str) -> Tuple[Dict[str, Optional[str]], float]:
        print(f"📋 SlipRuleParser.parse: Starting rule-based parsing")
        t = self._normalize(text)

        bank = self._extract_bank(t)
        print(f"📋 RULES: Bank extracted: {bank}")
        account = self._extract_account(t)
        print(f"📋 RULES: Account extracted: {account}")
        date_str = self._extract_date(t)
        print(f"📋 RULES: Date extracted: {date_str}")
        amount_num, amount_score = self._extract_amount(t)
        print(f"📋 RULES: Amount extracted: {amount_num} (score: {amount_score})")

        confidence = self._confidence(amount_num, date_str, bank, account, amount_score)
        result = {
            'amount': float(amount_num) if amount_num is not None else None,
            'date': date_str,
            'bank_name': bank,
            'account_number': account,
        }
        return result, confidence

    # Normalization
    def _normalize(self, text: str) -> str:
        t = text or ''
        t = t.replace('\u20b9', '₹')  # normalize rupee symbol
        t = re.sub(r"[•·●▪]", '-', t)
        t = re.sub(r"[_]{3,}|[-]{3,}", '\n', t)
        # unify spaces and lowercase for matching while keeping original for extraction where needed
        return t

    # Bank name
    def _extract_bank(self, text: str) -> Optional[str]:
        low = text.lower()
        for patt, canonical in self.bank_map.items():
            if re.search(patt, low, re.IGNORECASE):
                return canonical
        return None

    # Account number
    def _extract_account(self, text: str) -> Optional[str]:
        for patt in self.account_patterns:
            m = re.search(patt, text, re.IGNORECASE)
            if m:
                digits = re.sub(r"\D", '', m.group(1))
                if 6 <= len(digits) <= 20:
                    return digits
        # Fallback: a long 10-16 digit sequence nearby the word 'account'
        m2 = re.search(r"account[^\n]{0,40}?(\d{9,20})", text, re.IGNORECASE)
        if m2:
            return m2.group(1)
        return None

    # Date extraction
    def _extract_date(self, text: str) -> Optional[str]:
        for patt in self.date_patterns:
            m = re.search(patt, text, re.IGNORECASE)
            if not m:
                continue
            raw = m.group(1)
            normalized = self._standardize_date(raw)
            if normalized:
                return normalized
        return None

    def _standardize_date(self, date_str: str) -> Optional[str]:
        fmts = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%d/%m/%y', '%m/%d/%y', '%y/%m/%d', '%d-%m-%y', '%m-%d-%y', '%y-%m-%d'
        ]
        for f in fmts:
            try:
                dt = datetime.strptime(date_str, f)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    # Amount extraction
    def _extract_amount(self, text: str) -> Tuple[Optional[float], float]:
        low = text.lower()
        best_val: Optional[float] = None
        best_score = 0.0

        # 1) Amount numbers near labels
        for label in self.amount_labels:
            for m in re.finditer(label + r"[^\n]{0,40}?" + r"(₹|rs\.?|inr)?\s*(" + self.amount_number + r")",
                                  low, flags=re.IGNORECASE):
                num = self._to_number(m.group(2))
                if num is None:
                    continue
                score = 1.0
                if m.group(1):
                    score += 0.2  # currency marker helpful
                if 'total' in m.group(0):
                    score += 0.3
                if 0.01 <= num <= 10000000:
                    if score > best_score:
                        best_val, best_score = num, score

        # 2) Generic currency-prefixed amounts
        if best_val is None:
            for curr in self.currency_markers:
                for m in re.finditer(curr + r"\s*(" + self.amount_number + r")", low, flags=re.IGNORECASE):
                    num = self._to_number(m.group(1))
                    if num is None:
                        continue
                    if 0.01 <= num <= 10000000 and 0.8 > best_score:
                        best_val, best_score = num, 0.8

        # 3) Amount in words line
        if best_val is None:
            m = re.search(r"total\s*amount.*?in\s*words[^\n]*", low, flags=re.IGNORECASE)
            if m:
                wval = self._words_to_num(m.group(0))
                if wval is not None:
                    best_val, best_score = float(wval), 0.7
            else:
                # Look for any line that contains 'in words'
                for line in low.splitlines():
                    if 'in words' in line:
                        wval = self._words_to_num(line)
                        if wval is not None:
                            best_val, best_score = float(wval), 0.6
                            break

        return best_val, best_score

    def _to_number(self, s: str) -> Optional[float]:
        try:
            s = s.replace(',', '')
            return float(s)
        except Exception:
            return None

    # Basic amount-in-words parser for common cases
    def _words_to_num(self, line: str) -> Optional[int]:
        # e.g., 'Thirty Thousand', 'Thirty Thousand Only'
        tokens = re.findall(r"[a-zA-Z]+", line.lower())
        if not tokens:
            return None
        total = 0
        current = 0
        found = False
        for tok in tokens:
            if tok not in self.number_words:
                continue
            found = True
            val = self.number_words[tok]
            if val in (100, 1000, 100000, 1000000):
                if current == 0:
                    current = 1
                current *= val
                total += current
                current = 0
            else:
                current += val
        total += current
        if found and total > 0:
            return total
        return None

    # Confidence
    def _confidence(
        self,
        amount: Optional[float],
        date: Optional[str],
        bank: Optional[str],
        account: Optional[str],
        amount_score: float,
    ) -> float:
        score = 0.0
        if amount is not None:
            score += 0.45 + min(amount_score, 0.3) * 0.2
        if date:
            score += 0.25
        if bank:
            score += 0.1
        if account:
            score += 0.15
        # small boost if two or more fields present
        present = sum(1 for x in [amount, date, bank, account] if x)
        if present >= 2:
            score += 0.05
        return min(score, 1.0)


