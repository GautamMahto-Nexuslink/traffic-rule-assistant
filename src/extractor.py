import os
import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: str, output_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- Page {page_num} ---\n{text}")

    doc.close()

    full_text = "\n".join(pages_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[extractor] Extracted {len(full_text):,} characters from {len(pages_text)} pages → {output_path}")
    return full_text
