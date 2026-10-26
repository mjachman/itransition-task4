# iTransition Task 4 (Data Engineering Group)

Streamlit dashboard: https://itransition-task4.streamlit.app/

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY="your_key_here"
```

or as an environment variable:

```bash
export GEMINI_API_KEY="your_key_here"   # macOS/Linux
# setx GEMINI_API_KEY "your_key_here"   # Windows
```
