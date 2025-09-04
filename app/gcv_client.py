import base64
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import asyncio

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


class GCVClient:
    """Lightweight client for Google Cloud Vision REST API (v1) using API key.

    Uses the images:annotate endpoint for TEXT_DETECTION / DOCUMENT_TEXT_DETECTION.
    Docs: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate
    """

    def __init__(self, api_key: Optional[str] = None, default_feature: str = "DOCUMENT_TEXT_DETECTION") -> None:
        if load_dotenv is not None:
            try:
                load_dotenv()
            except Exception:
                pass
        self.api_key: Optional[str] = api_key or os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        self.endpoint: str = "https://vision.googleapis.com/v1/images:annotate"
        # Allowed: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION
        self.default_feature: str = default_feature

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_CLOUD_VISION_API_KEY not configured. Set env var or pass to GCVClient(api_key=...)."
            )
        return self.api_key

    @staticmethod
    def _cv_image_to_bytes(image: np.ndarray, format: str = "png") -> bytes:
        import cv2

        ext = ".png" if format.lower() == "png" else ".jpg"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise ValueError("Failed to encode image for Vision API request")
        return buf.tobytes()

    @staticmethod
    def _build_request_payload(
        image_bytes: bytes,
        feature_type: str,
        language_hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        request: Dict[str, Any] = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": feature_type}],
                }
            ]
        }
        if language_hints:
            request["requests"][0]["imageContext"] = {"languageHints": language_hints}
        return request

    @staticmethod
    def _extract_text_and_confidence(resp_json: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        """Return (text, confidence) from annotate response.

        Prefers fullTextAnnotation.text. Confidence is best-effort from pages if present.
        """
        try:
            responses = resp_json.get("responses", [])
            if not responses:
                return "", None
            r0 = responses[0]
            if "fullTextAnnotation" in r0 and r0["fullTextAnnotation"].get("text"):
                text = r0["fullTextAnnotation"]["text"] or ""
                conf_values: List[float] = []
                try:
                    pages = r0.get("fullTextAnnotation", {}).get("pages", [])
                    for p in pages:
                        blocks = p.get("blocks", [])
                        for b in blocks:
                            if "confidence" in b and isinstance(b["confidence"], (int, float)):
                                conf_values.append(float(b["confidence"]))
                except Exception:
                    pass
                avg_conf = sum(conf_values) / len(conf_values) if conf_values else None
                return text.strip(), avg_conf
            # Fallback to textAnnotations list
            if "textAnnotations" in r0 and r0["textAnnotations"]:
                parts = [ta.get("description", "") for ta in r0["textAnnotations"]]
                text = parts[0] if parts else ""
                return text.strip(), None
            return "", None
        except Exception:
            return "", None

    async def detect_text_from_bytes(
        self,
        image_bytes: bytes,
        feature_type: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[str, Optional[float]]:
        key = self._require_api_key()
        ft = (feature_type or self.default_feature).upper()
        payload = self._build_request_payload(image_bytes, ft, language_hints)
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{self.endpoint}?key={key}"
            res = await client.post(url, json=payload)
            if res.status_code >= 400:
                # Try to extract a helpful error message from response JSON
                err_detail = None
                try:
                    j = res.json()
                    err_detail = j.get("error", {}).get("message") or j
                except Exception:
                    err_detail = res.text
                if debug:
                    try:
                        print("[GCV] ERROR", res.status_code, err_detail)
                    except Exception:
                        pass
                raise RuntimeError(f"GCV error {res.status_code}: {err_detail}")
            data = res.json()
            text, conf = self._extract_text_and_confidence(data)
            if debug:
                try:
                    print("[GCV] POST", url)
                    print("[GCV] feature_type:", ft)
                    if language_hints:
                        print("[GCV] language_hints:", language_hints)
                    preview = (text or "")[:500]
                    print("[GCV] extracted_text_preview (500 chars):\n" + preview)
                    print("[GCV] confidence:", conf)
                except Exception:
                    pass
            return text, conf

    async def detect_text_from_cv2(
        self,
        image: np.ndarray,
        feature_type: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[str, Optional[float]]:
        img_bytes = self._cv_image_to_bytes(image, format="png")
        return await self.detect_text_from_bytes(
            img_bytes, feature_type=feature_type, language_hints=language_hints, timeout=timeout, debug=debug
        )

    async def detect_text_from_path(
        self,
        path: str,
        feature_type: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[str, Optional[float]]:
        with open(path, "rb") as f:
            img_bytes = f.read()
        return await self.detect_text_from_bytes(
            img_bytes, feature_type=feature_type, language_hints=language_hints, timeout=timeout, debug=debug
        )

    async def detect_text_best_from_bytes(
        self,
        image_bytes: bytes,
        language_hints: Optional[List[str]] = None,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[str, Optional[float]]:
        """Try DOCUMENT_TEXT_DETECTION and TEXT_DETECTION, pick best by confidence/length."""
        results: List[Tuple[str, Optional[float]]] = []
        for ft in ("DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"):
            try:
                t, c = await self.detect_text_from_bytes(
                    image_bytes, feature_type=ft, language_hints=language_hints, timeout=timeout, debug=debug
                )
                results.append((t, c))
            except Exception:
                results.append(("", None))
        # Rank by confidence then by length
        def score(rc: Tuple[str, Optional[float]]) -> Tuple[float, int]:
            txt, conf = rc
            return (float(conf) if conf is not None else 0.0, len(txt or ""))
        best = max(results, key=score)
        return best

    async def detect_text_best_from_path(
        self,
        path: str,
        language_hints: Optional[List[str]] = None,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[str, Optional[float]]:
        with open(path, "rb") as f:
            image_bytes = f.read()
        return await self.detect_text_best_from_bytes(
            image_bytes, language_hints=language_hints, timeout=timeout, debug=debug
        )

    async def post_process_with_openai(
        self,
        text: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        debug: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Use OpenAI to normalize and structure OCR text into key fields.

        Returns (parsed_json_dict | None, raw_response | None)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, None
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None, None

        def _call_sync() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
            client = OpenAI()
            mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            prompt = (
                "You are a strict parser. Extract deposit slip fields as compact JSON with keys: "
                "amount (number), date (YYYY-MM-DD), bank_name (string), account_number (string). "
                "Infer missing separators and normalize OCR errors conservatively."
            )
            msg = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"OCR TEXT:\n{text}"},
            ]
            try:
                resp = client.chat.completions.create(
                    model=mdl,
                    messages=msg,
                    temperature=temperature,
                    timeout=timeout,
                )
                raw = resp.choices[0].message.content if resp and resp.choices else None
                parsed: Optional[Dict[str, Any]] = None
                if raw:
                    import json

                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        # try to find a JSON object substring
                        import re

                        m = re.search(r"\{[\s\S]*\}", raw)
                        if m:
                            try:
                                parsed = json.loads(m.group(0))
                            except Exception:
                                parsed = None
                if debug:
                    try:
                        print("[OpenAI] model:", mdl)
                        print("[OpenAI] raw_response:\n", raw)
                        print("[OpenAI] parsed:", parsed)
                    except Exception:
                        pass
                return parsed, raw
            except Exception:
                return None, None

        return await asyncio.to_thread(_call_sync)

if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python -m app.gcv_client <image_path> [TEXT_DETECTION|DOCUMENT_TEXT_DETECTION]")
        sys.exit(1)

    image_path = sys.argv[1]
    ftype = sys.argv[2] if len(sys.argv) > 2 else "DOCUMENT_TEXT_DETECTION"

    async def _main() -> None:
        client = GCVClient()
        text, conf = await client.detect_text_from_path(
            image_path,
            feature_type=ftype,
            language_hints=["en"],
            debug=True,
        )
        print("\n=== Extracted Text ===\n")
        print(text)
        print("\nConfidence:", conf)

    asyncio.run(_main())


