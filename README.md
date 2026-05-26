[README.md](https://github.com/user-attachments/files/28280293/README.md)
# Loan Approval Intelligence — Streamlit + Gemini

GA-optimised Fuzzy Logic loan risk engine + a Gemini AI agent that calls the
engine as a tool (observe → reason → call tools → explain → act).

## Files
- `streamlit_app.py`  — UI + Gemini tool-calling agent
- `fuzzy_engine.py`   — the FIS + GA engine (unchanged from the original project)
- `loan_data.csv`     — dataset the GA trains on
- `requirements.txt`  — dependencies

## Get a free Gemini key
1. Go to https://aistudio.google.com → "Get API key" → create key (free).

## Option A — Deploy online (no local setup, gives a public link)
1. Put these 4 files in a GitHub repo (public is fine).
2. Go to https://share.streamlit.io → New app → pick your repo →
   main file = `streamlit_app.py` → Deploy.
3. In the app's **Settings → Secrets**, paste:
       GEMINI_API_KEY = "your-key-here"
4. Share the public URL. Anyone can open it.

## Option B — Run locally
    pip install -r requirements.txt
    # put your key in .streamlit/secrets.toml  (see template)
    streamlit run streamlit_app.py

Note: the GA runs once on first load (cached afterwards), so the very first
open may take ~10-30s.
