
import re


KNOWN_HEADINGS = {
    "ABSTRACT",
    "KEYWORDS",
    "INTRODUCTION",
    "LITERATURE REVIEW",
    "RELATED WORK",
    "RESEARCH METHODOLOGY",
    "METHODOLOGY",
    "PROPOSED METHODOLOGY",
    "ARTIFICIAL INTELLIGENCE METHODS",
    "MACHINE LEARNING",
    "DEEP LEARNING",
    "NATURAL LANGUAGE PROCESSING",
    "AUTOMATION & ROBOTICS",
    "MACHINE VISION",
    "KNOWLEDGE-BASED SYSTEMS",
    "NEURAL NETWORKS",
    "APPLICATIONS OF AI",
    "SOME OTHER APPLICATIONS",
    "RESULTS",
    "DISCUSSION",
    "FUTURE OF AI",
    "CONCLUSION",
    "REFERENCES"
}


def detect_heading(line):

    cleaned = line.strip()

    # Remove numbering, such as "1. INTRODUCTION"
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)

    # Remove punctuation at the end
    cleaned = re.sub(r"[\s:–—-]+$", "", cleaned)

    # Remove trailing abbreviations in parentheses, such as (NLP)
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)

    cleaned = cleaned.strip()

    normalized = cleaned.upper()

    if normalized in KNOWN_HEADINGS:
        return normalized

    return None


def split_section_into_chunks(
    text,
    filename,
    page_number,
    section,
    chunk_size,
    overlap
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append({
            "filename": filename,
            "page_number": page_number,
            "section": section,
            "text": chunk_text
        })

        start += chunk_size - overlap

    return chunks


def create_chunks(
    extracted_pages,
    chunk_size=150,
    overlap=30
):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    chunks = []

    for page_data in extracted_pages:

        text = page_data["text"]

        filename = page_data["filename"]

        page_number = page_data["page_number"]

        lines = text.splitlines()

        current_section = "General Content"

        section_lines = []

        for line in lines:

            heading = detect_heading(line)

            if heading:

                # Process the previous section
                section_text = " ".join(section_lines).strip()

                if section_text:

                    section_chunks = split_section_into_chunks(
                        section_text,
                        filename,
                        page_number,
                        current_section,
                        chunk_size,
                        overlap
                    )

                    chunks.extend(section_chunks)

                # Start the newly detected section
                current_section = heading

                section_lines = []

            else:

                if line.strip():
                    section_lines.append(line.strip())

        # Process the final section on this page
        section_text = " ".join(section_lines).strip()

        if section_text:

            section_chunks = split_section_into_chunks(
                section_text,
                filename,
                page_number,
                current_section,
                chunk_size,
                overlap
            )

            chunks.extend(section_chunks)

    # Diagnostic output
    print("\n========== SECTION-AWARE CHUNKING ==========")

    print("Total extracted pages:", len(extracted_pages))

    print("Total chunks created:", len(chunks))

    for i, chunk in enumerate(chunks[:10]):

        print(f"\nChunk {i + 1}")

        print("Page:", chunk["page_number"])

        print("Section:", chunk["section"])

        print("Word count:", len(chunk["text"].split()))

        print("Text preview:", chunk["text"][:200])

    print("============================================\n")

    return chunks
