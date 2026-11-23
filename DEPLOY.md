# Deploying the prototype

Recommended: **Streamlit Community Cloud** (free, deploys from GitHub in a few minutes, no Docker).
Alternatives: Hugging Face Spaces (Docker) or Render (Docker blueprint included).

Whatever the host, put API keys in the host's **secrets / environment variables** only. Never commit
`.env` or `.streamlit/secrets.toml` (both are in `.gitignore`).

---

## Step 0: Get a Gemini API key (optional but recommended)

Without a key the app still works with the sample house and with manually drawn boxes.

1. Sign in at https://aistudio.google.com with your Google account.
2. **Get API key → Create API key**. Copy it somewhere safe (you will paste it into the host's secrets, not into code).
3. Decide on an access code (any phrase). Reviewers type it in the sidebar to use your key; others can paste
   their own key instead. The app caps hosted-key use at 12 AI calls per browser session.

## Step 1: Push the code to GitHub

The folder is already a git repository with one local commit. Nothing has been pushed.

1. Sign in at https://github.com and click **New repository**.
   - Name: e.g. `exterior-renovation-planner`
   - Visibility: **Public** is simplest for reviewers. Private also works with Streamlit Cloud once you
     grant access.
   - Do **not** add a README, .gitignore or licence (the repo already has them).
2. In PowerShell:

```powershell
cd C:\Users\LENOVO\Documents\E2M_AI_Assessment
```

```powershell
git remote add origin https://github.com/<your-username>/exterior-renovation-planner.git
```

```powershell
git push -u origin main
```

Git may open a browser window to sign in to GitHub the first time.

## Option A: Streamlit Community Cloud (recommended)

1. Go to https://share.streamlit.io and **Continue with GitHub** (authorise Streamlit).
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/exterior-renovation-planner`, Branch: `main`, Main file path: `app.py`.
4. Optional App URL: e.g. `e2m-renovation-planner` → `https://e2m-renovation-planner.streamlit.app`.
5. **Advanced settings**:
   - Python version: **3.13**.
   - Secrets: paste (with your real values):

     ```toml
     GEMINI_API_KEY = "your-key"
     APP_ACCESS_CODE = "your-access-code"
     ```

6. Click **Deploy**. The first build takes 3-5 minutes. Then open the URL, click **Load the sample house**, and
   with the access code entered try **Detect components with AI** on a real photo.

Notes: free apps sleep after a period of no traffic and wake on the next visit (about 30 s). Open the link
yourself shortly before sending it. Secrets can be changed later under **App → Settings → Secrets**.

## Option B: Hugging Face Spaces (Docker)

1. Sign in at https://huggingface.co, then **New Space**: name it, SDK **Docker** (blank template),
   hardware **CPU basic (free)**, visibility Public.
2. Spaces read their settings from the top of `README.md`. Add this block as the very first lines of
   `README.md` in the copy you push to the Space:

   ```yaml
   ---
   title: Exterior Renovation Planner
   emoji: 🏠
   colorFrom: yellow
   colorTo: gray
   sdk: docker
   app_port: 7860
   ---
   ```

3. Push the code to the Space's git remote (shown on the Space page). Hugging Face asks for a *write* access
   token as the git password (Settings → Access Tokens).
4. Space **Settings → Variables and secrets → New secret**: `GEMINI_API_KEY`, `APP_ACCESS_CODE`.
5. The Space builds from the `Dockerfile` and runs on port 7860.

## Option C: Render (Docker blueprint)

1. Push to GitHub (Step 1), sign in at https://render.com with GitHub.
2. **New → Blueprint**, pick the repo. Render reads `render.yaml` (one free Docker web service).
3. When prompted, enter values for `GEMINI_API_KEY` and `APP_ACCESS_CODE`.
4. Free instances sleep after inactivity; the first request takes about a minute.

## What to send reviewers

- Live link (the Streamlit / HF / Render URL)
- Access code (if you set `APP_ACCESS_CODE`), or a note that they can paste their own Gemini key in the sidebar
- GitHub repository link
- How to try it with no key: **1 · Upload → Load the sample house**

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No AI key available" on the hosted app | The secret name must be exactly `GEMINI_API_KEY`. Reboot the app after editing secrets. |
| Detection error mentioning 401/403 | Key wrong, or the Gemini API is not enabled for that key. |
| Detection error mentioning 429 | Free-tier quota hit. Wait a minute, or use another key. |
| Model not found | Set `GEMINI_MODEL` in secrets to a model your key can use (e.g. a newer Flash model). |
| Build fails on a package version | Lower the Python version to 3.12 in Advanced settings, or loosen the pin in `requirements.txt`. |
