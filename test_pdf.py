
import fitz

from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_chunks


# Create a temporary sample PDF
doc = fitz.open()

for i in range(3):

    page = doc.new_page()

    page.insert_text(
        (72, 72),
        f"This is sample research content for page {i + 1}. "
        * 100
    )

doc.save("sample_research.pdf")
doc.close()


# Open the sample PDF
with open("sample_research.pdf", "rb") as file:

    pages = extract_text_from_pdf(file)

# Generate chunks
chunks = create_chunks(
    pages,
    chunk_size=100,
    overlap=20
)


print("PDF PROCESSING RESULTS")

print("Total pages extracted:", len(pages))

print("Total chunks generated:", len(chunks))

print("\nFirst chunk:")

print(chunks[0])
