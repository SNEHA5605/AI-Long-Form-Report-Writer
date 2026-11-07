
import streamlit as st

from modules.prompt_chain import (
    generate_outline,
    save_chain_trace,
    execute_report_chain
)

from modules.pdf_processor import extract_text_from_pdf

from modules.text_chunker import create_chunks

from modules.source_retriever import (
    generate_embeddings,
    create_faiss_index,
    retrieve_sources
)

from modules.rag_answer import generate_rag_answer


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Report Writer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI Long-Form Report Writer")

st.write(
    "Generate, review and improve academic reports "
    "using Prompt Chaining, Semantic Search and AI."
)

st.divider()


# --------------------------------
# REPORT CONFIGURATION
# --------------------------------

st.header("1. Report Configuration")

topic = st.text_input(
    "Enter your report topic",
    placeholder="Example: Impact of AI on Education"
)

word_count = st.selectbox(
    "Target word count",
    [1000, 1500, 2000, 3000]
)

writing_style = st.selectbox(
    "Select writing style",
    [
        "Academic",
        "Professional",
        "Technical",
        "Informative"
    ]
)


# --------------------------------
# U3: UPLOAD RESEARCH DOCUMENTS
# --------------------------------

st.divider()

st.header("2. Upload Research Sources")

st.write(
    "Upload PDF research papers or documents. "
    "The system will extract text and prepare documents "
    "for semantic and keyword search."
)

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("Process Research Documents"):

    # Clear previous document data and results

    for key in [
        "source_pages",
        "source_chunks",
        "source_embeddings",
        "faiss_index",
        "retrieved_sources",
        "last_research_query",
        "rag_answer"
    ]:

        st.session_state.pop(key, None)

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        try:

            # STEP 1: EXTRACT PDF TEXT

            all_pages = []

            with st.spinner("Extracting text from PDFs..."):

                for uploaded_file in uploaded_files:

                    pages = extract_text_from_pdf(
                        uploaded_file
                    )

                    all_pages.extend(pages)

            if not all_pages:

                st.warning(
                    "No readable text was found in the uploaded PDFs. "
                    "The documents may be scanned PDFs."
                )

            else:

                # STEP 2: CREATE SECTION-AWARE CHUNKS

                with st.spinner(
                    "Creating section-aware text chunks..."
                ):

                    chunks = create_chunks(
                        all_pages,
                        chunk_size=250,
                        overlap=40
                    )

                if not chunks:

                    st.warning(
                        "No text chunks were created."
                    )

                else:

                    # STEP 3: ABSTRACT VERIFICATION

                    print(
                        "\n========== ABSTRACT VERIFICATION =========="
                    )

                    abstract_chunks = [
                        chunk for chunk in chunks
                        if chunk.get(
                            "section", ""
                        ).upper() == "ABSTRACT"
                    ]

                    print("Configured chunk size: 250")
                    print("Configured overlap: 40")

                    print(
                        "Number of Abstract chunks:",
                        len(abstract_chunks)
                    )

                    for i, chunk in enumerate(
                        abstract_chunks,
                        start=1
                    ):

                        print(f"\nAbstract Chunk {i}")

                        print(
                            "Filename:",
                            chunk["filename"]
                        )

                        print(
                            "Page:",
                            chunk["page_number"]
                        )

                        print(
                            "Section:",
                            chunk["section"]
                        )

                        print(
                            "Word count:",
                            len(chunk["text"].split())
                        )

                        print(
                            "Complete text:",
                            chunk["text"]
                        )

                    print(
                        "===========================================\n"
                    )

                    # STEP 4: GENERATE EMBEDDINGS AND FAISS INDEX

                    with st.spinner(
                        "Generating embeddings and building search index..."
                    ):

                        embeddings = generate_embeddings(
                            chunks
                        )

                        faiss_index = create_faiss_index(
                            embeddings
                        )

                    # STEP 5: STORE PROCESSED DATA

                    st.session_state["source_pages"] = all_pages

                    st.session_state["source_chunks"] = chunks

                    st.session_state["source_embeddings"] = embeddings

                    st.session_state["faiss_index"] = faiss_index

                    st.success(
                        "Research documents processed successfully!"
                    )

                    # PROCESSING SUMMARY

                    st.write("### Processing Summary")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "PDF pages",
                        len(all_pages)
                    )

                    col2.metric(
                        "Text chunks",
                        len(chunks)
                    )

                    col3.metric(
                        "Indexed passages",
                        faiss_index.ntotal
                    )

                    st.write("**Chunk size:** 250 words")

                    st.write("**Chunk overlap:** 40 words")

                    st.write(
                        "**Abstract chunks:**",
                        len(abstract_chunks)
                    )

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# --------------------------------
# U3: HYBRID SOURCE RETRIEVAL
# --------------------------------

if "faiss_index" in st.session_state:

    st.divider()

    st.header("3. Search Research Sources")

    st.write(
        "Hybrid Search combines semantic similarity, "
        "keyword matching and section relevance."
    )

    research_query = st.text_input(
        "Enter your research question",
        placeholder=(
            "Example: What is the main objective "
            "of this research paper?"
        )
    )

    available_chunks = st.session_state[
        "faiss_index"
    ].ntotal

    maximum_results = min(
        10,
        available_chunks
    )

    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=maximum_results,
        value=maximum_results
    )

    if st.button("Retrieve Relevant Sources"):

        if not research_query.strip():

            st.warning(
                "Please enter a research question."
            )

        else:

            try:

                with st.spinner(
                    "Searching research documents..."
                ):

                    results = retrieve_sources(
                        research_query,
                        st.session_state["source_chunks"],
                        st.session_state["faiss_index"],
                        top_k=top_k
                    )

                    st.session_state["retrieved_sources"] = results

                    st.session_state["last_research_query"] = (
                        research_query
                    )

                    # Clear answer from previous question

                    st.session_state.pop(
                        "rag_answer",
                        None
                    )

                st.success(
                    "Relevant sources retrieved successfully!"
                )

            except Exception as error:

                st.error(
                    f"Source retrieval failed: {error}"
                )


# --------------------------------
# DISPLAY RETRIEVED SOURCES
# --------------------------------

if "retrieved_sources" in st.session_state:

    st.divider()

    st.subheader("Retrieved Research Passages")

    for number, source in enumerate(
        st.session_state["retrieved_sources"],
        start=1
    ):

        section_name = source.get(
            "section",
            "Unknown"
        )

        with st.expander(
            f"Source {number}: "
            f"{source['filename']} | "
            f"Page {source['page_number']} | "
            f"{section_name}"
        ):

            st.write(
                f"**Section:** {section_name}"
            )

            st.write(
                f"**Page:** {source['page_number']}"
            )

            st.write(
                f"**Filename:** {source['filename']}"
            )

            st.divider()

            st.write("**Retrieved Text**")

            st.write(
                source["text"]
            )


# --------------------------------
# U3: RAG ANSWER GENERATION
# --------------------------------

if "retrieved_sources" in st.session_state:

    st.divider()

    st.header("4. Generate Research Answer")

    st.write(
        "The AI system will analyze the retrieved research passages "
        "and generate an answer using your uploaded documents."
    )

    st.write(
        "**Question:**",
        st.session_state["last_research_query"]
    )

    if st.button(
        "Generate Answer from Retrieved Sources",
        type="primary"
    ):

        try:

            with st.spinner(
                "Generating an evidence-based answer..."
            ):

                answer = generate_rag_answer(
                    st.session_state["last_research_query"],
                    st.session_state["retrieved_sources"]
                )

                st.session_state["rag_answer"] = answer

            st.success(
                "Answer generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Answer generation failed: {error}"
            )


# --------------------------------
# DISPLAY RAG ANSWER
# --------------------------------

if "rag_answer" in st.session_state:

    st.divider()

    st.subheader("AI-Generated Research Answer")

    st.markdown(
        st.session_state["rag_answer"]
    )

    # SOURCE REFERENCE KEY

    st.subheader("Source Reference Key")

    st.write(
        "Use these references to identify the passages "
        "cited in the generated answer."
    )

    for number, source in enumerate(
        st.session_state["retrieved_sources"],
        start=1
    ):

        st.write(
            f"**[S{number}]** "
            f"{source['filename']} | "
            f"Page {source['page_number']} | "
            f"{source.get('section', 'Unknown')}"
        )


# --------------------------------
# U1: GENERATE REPORT OUTLINE
# --------------------------------

st.divider()

st.header("5. Generate Report Outline")

if st.button(
    "Generate Report Outline",
    type="secondary"
):

    if not topic.strip():

        st.warning(
            "Please enter a report topic."
        )

    else:

        try:

            with st.spinner("Generating report outline..."):

                outline, trace = generate_outline(
                    topic,
                    word_count,
                    writing_style
                )

                save_chain_trace(trace)

                st.session_state["outline"] = outline

                st.session_state["trace"] = trace

                # Save report configuration

                st.session_state["topic"] = topic

                st.session_state["word_count"] = word_count

                st.session_state["writing_style"] = writing_style

                # Clear previous report results

                for key in [
                    "draft",
                    "feedback",
                    "revised_report",
                    "u2_traces"
                ]:

                    st.session_state.pop(key, None)

            st.success(
                "Outline generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Error: {error}"
            )


# --------------------------------
# DISPLAY GENERATED OUTLINE
# --------------------------------

if "outline" in st.session_state:

    st.markdown(
        "### Generated Report Outline"
    )

    st.markdown(
        st.session_state["outline"]
    )


    # --------------------------------
    # U2: DRAFT, REVIEW AND REVISE
    # --------------------------------

    st.divider()

    st.header("6. Generate and Review Report")

    st.write(
        "The AI system will generate a report, "
        "review its quality and revise it."
    )

    if st.button(
        "Generate, Review and Improve Report"
    ):

        try:

            with st.spinner(
                "Drafting, reviewing and revising your report..."
            ):

                result = execute_report_chain(
                    st.session_state["topic"],
                    st.session_state["word_count"],
                    st.session_state["writing_style"],
                    st.session_state["outline"]
                )

                st.session_state["draft"] = result["draft"]

                st.session_state["feedback"] = result["feedback"]

                st.session_state["revised_report"] = (
                    result["revised_report"]
                )

                st.session_state["u2_traces"] = result["traces"]

            st.success(
                "Report successfully generated and revised!"
            )

        except Exception as error:

            st.error(
                f"Report generation failed: {error}"
            )


# --------------------------------
# DISPLAY ORIGINAL DRAFT
# --------------------------------

if "draft" in st.session_state:

    st.divider()

    st.header("7. Original Report Draft")

    st.markdown(
        st.session_state["draft"]
    )


# --------------------------------
# DISPLAY SELF-REVIEW
# --------------------------------

if "feedback" in st.session_state:

    st.divider()

    st.header("8. AI Self-Review")

    with st.expander(
        "View review feedback",
        expanded=True
    ):

        st.markdown(
            st.session_state["feedback"]
        )


# --------------------------------
# DISPLAY REVISED REPORT
# --------------------------------

if "revised_report" in st.session_state:

    st.divider()

    st.header("9. Final Revised Report")

    st.markdown(
        st.session_state["revised_report"]
    )

    revised_word_count = len(
        st.session_state["revised_report"].split()
    )

    st.metric(
        "Revised report word count",
        revised_word_count
    )


# --------------------------------
# U2 PROMPT CHAIN TRACE
# --------------------------------

if "u2_traces" in st.session_state:

    st.divider()

    st.header("10. U2 Prompt Chain Trace")

    for trace_item in st.session_state["u2_traces"]:

        with st.expander(
            f"Step {trace_item['step']}: "
            f"{trace_item['stage']}"
        ):

            st.json(trace_item)


# --------------------------------
# U1 OUTLINE GENERATION TRACE
# --------------------------------

if "trace" in st.session_state:

    with st.expander(
        "View U1 Outline Generation Trace"
    ):

        st.json(
            st.session_state["trace"]
        )
