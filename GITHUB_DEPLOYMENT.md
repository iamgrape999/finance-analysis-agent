# GitHub + Render Deployment

This project has two deployable parts:

1. FastAPI backend: deploy to Render using `render.yaml`.
2. Static browser frontend: deploy to GitHub Pages using `.github/workflows/pages.yml`.

GitHub Pages cannot run Python or store private API keys. The browser UI on GitHub Pages must call the Render backend URL.

## 1. Push To GitHub

Install Git if needed:

```powershell
winget install Git.Git
```

Restart PowerShell, then:

```powershell
cd "C:\Users\HAN-LI CHANG\Documents\Codex\2026-04-19-colab-python-3-github-ai-writefile"
git init
git add .
git commit -m "Deploy finance analysis agent web app"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/finance-analysis-agent.git
git push -u origin main
```

Do not commit `.env`; `.gitignore` excludes it.

## 2. Deploy Backend On Render

In Render:

1. New + -> Blueprint.
2. Select the GitHub repository.
3. Render reads `render.yaml`.
4. Add environment secrets:

```env
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
CF_API_TOKEN=...
CF_ACCOUNT_ID=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

At least one provider key is required.

On Render free tier, persistent disks are not supported. The included `render.yaml` therefore uses temporary storage:

```text
/tmp/memory
```

This is enough for deployment testing, but memory may disappear after restarts or redeploys. For persistent memory, upgrade Render and add a disk mounted at `/var/data`, then set:

```env
MEMORY_ROOT=/var/data/memory
```

## 3. Enable GitHub Pages

In GitHub:

1. Repository -> Settings -> Pages.
2. Source: GitHub Actions.
3. Push to `main`; the workflow deploys the `static/` folder.

## 4. Connect GitHub Pages Frontend To Render Backend

Open the GitHub Pages URL.

In the left settings panel, set:

```text
Backend API URL = https://YOUR-RENDER-SERVICE.onrender.com
```

The value is stored in browser `localStorage`.

Then check:

```text
/api/health
```

from the UI status. A provider list should appear.

## 5. Test

Ask:

```text
hi，請只用一句繁體中文回覆。
```

The response should show:

```text
platform=...
model=...
tokens=...
time=...
```
