from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

import google.generativeai as genai
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gemini Synthetic Persona Lab", layout="wide")

PERSONAS = [
    {"id": "mum", "name": "Sarah M.", "label": "Health-First Mum", "country": "AU"},
    {"id": "senior", "name": "Raj P.", "label": "Budget Senior", "country": "AU"},
    {"id": "exec", "name": "Marcus B.", "label": "Busy Executive", "country": "US"},
    {"id": "genz", "name": "Aisha K.", "label": "Gen Z Ingredient Hunter", "country": "UK"},
    {"id": "prag", "name": "David L.", "label": "Skeptical Pragmatist", "country": "US"},
]

SUBCATEGORIES = {
    "Oral Health": ["Daily toothpaste", "Whitening", "Sensitivity", "Mouthwash / rinse"],
    "OTC": ["Pain relief / analgesics", "Cold & flu", "Digestive", "Allergy / hay fever"],
    "Wellness": ["Adult multivitamins", "Immunity", "Sleep & mood", "Sports nutrition"],
}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")  # type: ignore[attr-defined]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_RE.search(cleaned)
        if not match:
            raise RuntimeError("Model output did not contain valid JSON.")
        return json.loads(match.group(0))


def build_prompts(persona: dict, category: str, subcategory: str, subject: str) -> tuple[str, str]:
    system = (
        f"You are {persona['name']} ({persona['label']}) from {persona['country']}. "
        "Respond as a real consumer in first person. Return JSON only."
    )
    user = f"""
Evaluate this claim/name for {category} -> {subcategory}.

Subject: "{subject}"

Return JSON with:
{{
  "scores": {{
    "believability": 1-7,
    "relevance": 1-7,
    "clarity": 1-7,
    "differentiation": 1-7,
    "purchase_intent": 1-7
  }},
  "verbatim": "2 short sentences",
  "top_positive": "one sentence",
  "top_concern": "one sentence"
}}
"""
    return system, user


def evaluate_persona(model_name: str, persona: dict, category: str, subcategory: str, subject: str) -> dict:
    system, user = build_prompts(persona, category, subcategory, subject)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system)
    response = model.generate_content(user)
    payload = parse_json(response.text or "")
    scores = payload["scores"]
    overall = round(mean([scores["believability"], scores["relevance"], scores["clarity"], scores["differentiation"], scores["purchase_intent"]]), 2)
    return {
        "persona": persona["name"],
        "segment": persona["label"],
        "country": persona["country"],
        "overall": overall,
        **scores,
        "verbatim": payload.get("verbatim", ""),
        "top_positive": payload.get("top_positive", ""),
        "top_concern": payload.get("top_concern", ""),
    }


def run_batch(model_name: str, personas: list[dict], category: str, subcategory: str, subject: str) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=min(6, len(personas))) as pool:
        futures = [pool.submit(evaluate_persona, model_name, p, category, subcategory, subject) for p in personas]
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda r: r["persona"])
    return rows


def main() -> None:
    st.title("Gemini Synthetic Persona Lab")
    st.caption("Backend-managed Gemini key only. No user API key input.")

    api_key = get_api_key()
    if not api_key:
        st.error("Missing GEMINI_API_KEY in secrets or environment.")
        st.stop()
    genai.configure(api_key=api_key)

    st.sidebar.header("Setup")
    model_name = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    countries = st.sidebar.multiselect("Country filter", ["AU", "US", "UK"], default=["AU", "US", "UK"])
    selected_personas = [p for p in PERSONAS if p["country"] in countries]

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", list(SUBCATEGORIES.keys()))
    with col2:
        subcategory = st.selectbox("Sub-category", SUBCATEGORIES[category])

    subject = st.text_area("Claim or name", placeholder="Type claim or product name...")
    run = st.button("Run test", type="primary", use_container_width=True)

    if run:
        if not subject.strip():
            st.warning("Enter a claim or name first.")
            st.stop()
        if not selected_personas:
            st.warning("Select at least one country/persona.")
            st.stop()

        with st.spinner("Running persona evaluations..."):
            rows = run_batch(model_name, selected_personas, category, subcategory, subject.strip())

        df = pd.DataFrame(rows)
        st.subheader("Results")
        st.metric("Overall mean", round(df["overall"].mean(), 2))
        st.dataframe(df.drop(columns=["verbatim", "top_positive", "top_concern"]), use_container_width=True)

        st.subheader("Verbatims")
        for row in rows:
            with st.expander(f"{row['persona']} — {row['segment']} ({row['country']})"):
                st.write(row["verbatim"])
                st.write(f"Top positive: {row['top_positive']}")
                st.write(f"Top concern: {row['top_concern']}")

        st.download_button(
            "Download results JSON",
            data=json.dumps(rows, indent=2).encode("utf-8"),
            file_name="gemini_persona_results.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
