
import pymupdf


def extract_text_from_pdf(uploaded_file):

    extracted_pages = []

    pdf_bytes = uploaded_file.read()

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:

        for page_number, page in enumerate(document):

            text = page.get_text("text").strip()

            if not text:
                continue

            page_data = {
                "filename": uploaded_file.name,
                "page_number": page_number + 1,
                "text": text
            }

            extracted_pages.append(page_data)

    finally:

        document.close()

    return extracted_pages
