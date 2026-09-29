# Deployment guide

The app needs no secrets and no database, so deployment is simple.

## Option 1: Streamlit Community Cloud (recommended, free)

1. Create a public GitHub repository and push this project to it:
   ```bash
   git init
   git add .
   git commit -m "DemandPilot: demand forecasting and inventory decision assistant"
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/demandpilot.git
   git push -u origin main
   ```
   Before pushing, run `git status` and check that no `.env` file or private data is listed.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings**, choose Python 3.11 or 3.12.
5. Click **Deploy**. The first build takes a few minutes because LightGBM and statsmodels are installed.
6. Copy the app URL into the README and your CV.

Optional: to enable the language-model rephrasing, add these under the app's **Secrets** and add `anthropic` to `requirements.txt`:
```toml
DEMANDPILOT_LLM_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = "your key"
```
Never put the key in the repository.

Free apps go to sleep after a period without visitors. Open the link a few minutes before an interview so it has time to wake up.

## Option 2: Run locally (always works as a fallback)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

## Option 3: Docker

```bash
docker build -t demandpilot .
docker run -p 8501:8501 demandpilot
```
Then open http://localhost:8501.

## Checks before sharing the link

- [ ] App loads with sample data and every tab opens without an error
- [ ] Changing lead time changes the recommendation
- [ ] Downloads work
- [ ] README screenshots and links display on GitHub
- [ ] `pytest` passes
- [ ] No secrets, `.env` files or real customer data in the repository
