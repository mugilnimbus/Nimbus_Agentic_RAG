import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - only used if the runtime lacks pypdf.
    PdfReader = None


IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
}


def document_text_from_payload(payload: dict, rag, extraction_workers: int) -> tuple[str, str, bool]:
    file_type = str(payload.get("file_type") or "text").lower()
    should_distill = bool(payload.get("distill"))
    if file_type == "image":
        mime_type = str(payload.get("mime_type") or "").lower()
        if mime_type not in IMAGE_MIME_TYPES:
            raise ValueError("Unsupported image type.")
        encoded = str(payload.get("file_data") or "")
        validate_base64(encoded, "Image")
        notes = rag.extract_image_notes(
            str(payload.get("name") or "Uploaded image"),
            encoded,
            mime_type,
        )
        return notes, "raw", True

    if file_type != "pdf":
        return str(payload.get("text") or ""), "raw", should_distill

    return extract_pdf_text(payload, extraction_workers), "raw", should_distill


def extract_pdf_text(payload: dict, extraction_workers: int) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF support requires pypdf, but it is not installed.")

    encoded = str(payload.get("file_data") or "")
    if not encoded:
        raise ValueError("PDF file data is required.")

    pdf_bytes = validate_base64(encoded, "PDF")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)
    pages_by_index: dict[int, str] = {}
    workers = max(1, min(extraction_workers, page_count or 1))

    if workers == 1:
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_by_index[index] = f"Page {index}\n{text}"
    else:
        page_batches = partition_pages(list(range(1, page_count + 1)), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(extract_pdf_page_batch, pdf_bytes, batch): batch
                for batch in page_batches
                if batch
            }
            for future in as_completed(futures):
                for index, text in future.result():
                    if text.strip():
                        pages_by_index[index] = f"Page {index}\n{text}"

    extracted = "\n\n".join(
        pages_by_index[index] for index in sorted(pages_by_index)
    ).strip()
    if not extracted:
        raise ValueError("No extractable text was found in this PDF.")
    return extracted


def partition_pages(page_numbers: list[int], workers: int) -> list[list[int]]:
    batch_size = max(1, (len(page_numbers) + workers - 1) // workers)
    return [
        page_numbers[start : start + batch_size]
        for start in range(0, len(page_numbers), batch_size)
    ]


def extract_pdf_page_batch(pdf_bytes: bytes, page_numbers: list[int]) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = []
    for index in page_numbers:
        try:
            text = reader.pages[index - 1].extract_text() or ""
        except Exception as exc:
            text = f"[Page {index} text extraction failed: {exc}]"
        extracted.append((index, text))
    return extracted


def validate_base64(encoded: str, label: str) -> bytes:
    if not encoded:
        raise ValueError(f"{label} file data is required.")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"{label} file data is not valid base64.") from exc
