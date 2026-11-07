

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

from config import (
    MODEL_NAME,
    CHAIN_TRACE_FILE
)

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# FUNCTION 1: COMMUNICATE WITH GEMINI
# --------------------------------------------------

def call_gemini(prompt):

    # First check environment variables
    api_key = os.getenv("GEMINI_API_KEY")

    # If not found, check Streamlit Secrets
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. "
            "Please configure GEMINI_API_KEY in Streamlit Secrets."
        )

    # Create Gemini client
    client = genai.Client(api_key=api_key)

    # Send prompt to Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2500
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


# Temporary compatibility wrapper:
# Existing chain functions can continue using call_ollama().
def call_ollama(prompt):
    return call_gemini(prompt)


# --------------------------------------------------
# FUNCTION 2: LOAD PROMPT TEMPLATE
# --------------------------------------------------

def load_prompt(filename):

    prompt_path = BASE_DIR / "prompts" / filename

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------
# FUNCTION 3: SAVE CHAIN TRACE
# --------------------------------------------------

def save_chain_trace(trace):

    trace_path = BASE_DIR / CHAIN_TRACE_FILE

    trace_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if trace_path.exists():

        with open(trace_path, "r", encoding="utf-8") as file:
            try:
                traces = json.load(file)
            except json.JSONDecodeError:
                traces = []

    else:
        traces = []

    if not isinstance(traces, list):
        traces = []

    traces.append(trace)

    with open(trace_path, "w", encoding="utf-8") as file:

        json.dump(
            traces,
            file,
            indent=4,
            ensure_ascii=False
        )


# --------------------------------------------------
# FUNCTION 4: CREATE TRACE RECORD
# --------------------------------------------------

def create_trace(step, stage, prompt, output):

    return {
        "step": step,
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "prompt": prompt,
        "output": output
    }


# --------------------------------------------------
# FUNCTION 5: GENERATE REPORT OUTLINE (U1)
# --------------------------------------------------

def generate_outline(topic, word_count, writing_style):

    prompt_template = load_prompt("outline_prompt.txt")

    prompt = (
        prompt_template
        .replace("{topic}", topic)
        .replace("{word_count}", str(word_count))
        .replace("{writing_style}", writing_style)
    )

    outline = call_ollama(prompt)

    trace = create_trace(
        1,
        "Outline Generation",
        prompt,
        outline
    )

    trace["input"] = {
        "topic": topic,
        "word_count": word_count,
        "writing_style": writing_style
    }

    return outline, trace


# --------------------------------------------------
# FUNCTION 6: GENERATE COMPLETE REPORT (U2)
# --------------------------------------------------

def generate_draft(topic, word_count, writing_style, outline):

    prompt_template = load_prompt("writing_prompt.txt")

    prompt = (
        prompt_template
        .replace("{topic}", topic)
        .replace("{word_count}", str(word_count))
        .replace("{writing_style}", writing_style)
        .replace("{outline}", outline)
    )

    draft = call_ollama(prompt)

    trace = create_trace(
        2,
        "Report Drafting",
        prompt,
        draft
    )

    return draft, trace


# --------------------------------------------------
# FUNCTION 7: REVIEW REPORT (U2)
# --------------------------------------------------

def review_report(topic, draft):

    prompt_template = load_prompt("review_prompt.txt")

    prompt = (
        prompt_template
        .replace("{topic}", topic)
        .replace("{draft}", draft)
    )

    feedback = call_ollama(prompt)

    trace = create_trace(
        3,
        "Self Review",
        prompt,
        feedback
    )

    return feedback, trace


# --------------------------------------------------
# FUNCTION 8: REVISE REPORT (U2)
# --------------------------------------------------

def revise_report(draft, feedback):

    prompt_template = load_prompt("revision_prompt.txt")

    prompt = (
        prompt_template
        .replace("{draft}", draft)
        .replace("{feedback}", feedback)
    )

    revised_report = call_ollama(prompt)

    trace = create_trace(
        4,
        "Report Revision",
        prompt,
        revised_report
    )

    return revised_report, trace


# --------------------------------------------------
# FUNCTION 9: EXECUTE COMPLETE U2 CHAIN
# --------------------------------------------------

def execute_report_chain(
    topic,
    word_count,
    writing_style,
    outline
):

    # Stage 1: Generate report
    draft, draft_trace = generate_draft(
        topic,
        word_count,
        writing_style,
        outline
    )

    # Stage 2: Review report
    feedback, review_trace = review_report(
        topic,
        draft
    )

    # Stage 3: Revise report
    revised_report, revision_trace = revise_report(
        draft,
        feedback
    )

    # Save every stage in the execution history
    save_chain_trace(draft_trace)
    save_chain_trace(review_trace)
    save_chain_trace(revision_trace)

    return {
        "draft": draft,
        "feedback": feedback,
        "revised_report": revised_report,
        "traces": [
            draft_trace,
            review_trace,
            revision_trace
        ]
    }
