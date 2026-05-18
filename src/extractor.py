import os

import fitz  # pymupdf


def _extract_with_pymupdf(doc: fitz.Document) -> str:
    """Standard text extraction — works for PDFs with embedded text."""
    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- Page {page_num} ---\n{text}")
    return "\n".join(pages_text)


def _extract_with_ocr(pdf_path: str) -> str:
    """
    OCR fallback for scanned PDFs or PDFs where text is stored as vector paths
    (e.g. exported from CorelDRAW / Illustrator).
    Requires: tesseract, pdf2image, pytesseract, Pillow.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "OCR dependencies missing. Run: uv pip install pytesseract pdf2image Pillow"
        )

    print(f"[extractor] Falling back to OCR (vector/scanned PDF detected)…")
    images = convert_from_path(pdf_path, dpi=200)
    pages_text = []
    for page_num, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, lang="eng")
        if text.strip():
            pages_text.append(f"--- Page {page_num} ---\n{text}")
        if page_num % 10 == 0:
            print(f"[extractor] OCR progress: {page_num}/{len(images)} pages…")
    return "\n".join(pages_text)


def extract_text_from_pdf(pdf_path: str, output_path: str) -> str:
    """
    Extract text from a PDF file and save to output_path.
    Automatically uses OCR if direct text extraction yields nothing
    (handles scanned PDFs and vector-text PDFs from design tools).
    """
    doc = fitz.open(pdf_path)
    is_encrypted = doc.is_encrypted
    doc.close()

    if is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be read: {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = _extract_with_pymupdf(doc)
    doc.close()

    if not full_text.strip():
        full_text = _extract_with_ocr(pdf_path)

    if not full_text.strip():
        print(f"[extractor] WARNING: No text could be extracted from {os.path.basename(pdf_path)}")
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    method = "OCR" if "--- Page 1 ---" not in full_text[:50] else "direct"
    page_count = full_text.count("--- Page ")
    print(f"[extractor] {os.path.basename(pdf_path)}: {len(full_text):,} chars, {page_count} pages ({method}) → {output_path}")
    return full_text
