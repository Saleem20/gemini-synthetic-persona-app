# Gemini Synthetic Persona App

New standalone Streamlit app (separate from your existing repo) using Gemini API with backend-managed key.

## Run locally

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and set:

```toml
GEMINI_API_KEY = "your-key"
```

3. Start app:

```bash
streamlit run app.py
```

## Notes

- No user-facing API key input.
- Country filter and category/sub-category selectors included.
- Results include scores and verbatims, with JSON export.
