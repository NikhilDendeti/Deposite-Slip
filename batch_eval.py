import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def find_images(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
    files: List[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts and p.stat().st_size > 1024:  # skip tiny placeholders
            files.append(p)
    return files


async def process_one_image(processor, image_path: Path, modes: List[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "image": str(image_path),
        "size_bytes": image_path.stat().st_size if image_path.exists() else None,
        "results": {}
    }
    for mode in modes:
        try:
            result = await processor.process_deposit_slip(str(image_path), mode=mode)
            record["results"][mode] = result
        except Exception as e:
            record["results"][mode] = {"error": str(e)}
    return record


async def bounded_gather(tasks: List, limit: int = 2) -> List:
    sem = asyncio.Semaphore(limit)

    async def run_coro(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(run_coro(t) for t in tasks))


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluate deposit slip OCR modes on uploads directory.")
    parser.add_argument("--uploads-dir", default="uploads", help="Directory containing images")
    parser.add_argument("--output", default="outputs/results.jsonl", help="Path to write JSONL results")
    parser.add_argument("--modes", nargs="*", default=["ocr", "llm", "vision"], help="Processing modes to run")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent images")
    args = parser.parse_args()

    uploads_dir = Path(args.uploads_dir)
    if not uploads_dir.exists():
        raise SystemExit(f"Uploads directory not found: {uploads_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import after sys.path is set by runner
    from app.ocr_service import OCRProcessor  # type: ignore

    processor = OCRProcessor()
    images = find_images(uploads_dir)
    if not images:
        print(f"No images found in {uploads_dir}")
        return

    print(f"Found {len(images)} images. Modes: {args.modes}. Writing to {output_path}")

    # Prepare tasks
    tasks = [process_one_image(processor, img, args.modes) for img in images]
    results = await bounded_gather(tasks, limit=max(1, args.concurrency))

    # Write JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also write a compact summary JSON for quick inspection
    summary: List[Dict[str, Any]] = []
    for rec in results:
        row: Dict[str, Any] = {"image": rec["image"]}
        for mode, res in rec.get("results", {}).items():
            if isinstance(res, dict):
                row[f"{mode}_amount"] = res.get("amount")
                row[f"{mode}_date"] = res.get("date")
                row[f"{mode}_bank"] = res.get("bank_name")
                row[f"{mode}_account"] = res.get("account_number")
                row[f"{mode}_confidence"] = res.get("confidence")
            else:
                row[f"{mode}_error"] = str(res)
        summary.append(row)

    with (output_path.parent / "summary.json").open("w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} records to {output_path}")
    print(f"Wrote summary to {(output_path.parent / 'summary.json')}")


if __name__ == "__main__":
    # Ensure project root on sys.path
    import sys
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    # Also add app root if needed
    app_dir = here / "app"
    if app_dir.exists() and str(here) not in sys.path:
        sys.path.insert(0, str(here))
    asyncio.run(main())


