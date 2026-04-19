# CathyChang AI Web

這個專案把原本 Colab 取向的 `Finance_Analysis_Agent_V581.py` 改成可部署的瀏覽器 AI 對話程式：

- GitHub repo 保存程式碼
- FastAPI 提供 `/api/chat`
- `static/` 提供瀏覽器聊天介面
- Render 使用 `render.yaml` 一鍵部署
- secrets 走環境變數
- `MEMORY_ROOT` 可掛 Render persistent disk

## 本機啟動

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

把 `.env` 裡至少一組模型金鑰補上，例如：

```env
APP_PASSWORD=choose-a-strong-password
OPENROUTER_API_KEY=sk-or-...
```

或使用 Fireworks AI：

```env
FIREWORKS_API_KEY=fw_...
FIREWORKS_MODEL=accounts/fireworks/models/minimax-m2p7
```

啟動：

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

打開：

```text
http://localhost:8000
```

健康檢查：

```text
http://localhost:8000/api/health
```

## API

### POST `/api/chat`

```json
{
  "thread_id": "demo-user-001",
  "message": "請分析台灣法金業務導入 AI 的機會與風險",
  "max_tokens": 4096,
  "temperature": 0.2,
  "use_history": true,
  "provider_order": "openrouter,gemini,cloudflare,groq,aws",
  "model_overrides": {
    "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
    "fireworks": "accounts/fireworks/models/minimax-m2p7"
  }
}
```

回應：

```json
{
  "thread_id": "demo-user-001",
  "reply": "..."
}
```

### GET `/api/threads/{thread_id}/messages`

讀取該 thread 的 JSONL 對話紀錄。

## Render 部署

1. 將 repo 推到 GitHub。
2. 在 Render 選 `New +` -> `Blueprint`。
3. 指向這個 GitHub repo。
4. Render 會讀取 `render.yaml` 建立 web service 與 persistent disk。
5. 到 Render service 的 Environment 補上至少一組模型金鑰。

建議最小設定：

```env
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
MEMORY_ROOT=/var/data/memory
AGENT_PROVIDER_ORDER=openrouter,fireworks,gemini,cloudflare,groq,aws
```

免費 Render service 不支援 persistent disk，所以 `render.yaml` 預設使用暫存路徑：

```text
/tmp/memory
```

這足以讓瀏覽器對話程式上線測試，但 Render 休眠、重部署或重啟後，檔案型記憶可能消失。若要永久保存對話記憶，請升級 Render 並加 persistent disk，再把 `MEMORY_ROOT` 改成 `/var/data/memory`；或改用 Postgres/SQLite on persistent storage。

## GitHub Pages 分離前端

目前 FastAPI 直接 serve `static/`，Render URL 可直接使用瀏覽器介面。

如果你之後要改成 GitHub Pages 當純前端，請把 `static/app.js` 的 fetch URL 改成 Render 後端網址，例如：

```js
const API_BASE = "https://finance-analysis-agent.onrender.com";
fetch(`${API_BASE}/api/chat`, ...)
```

同時 Render 設定：

```env
CORS_ALLOW_ORIGINS=https://YOUR_NAME.github.io
```

## 安全提醒

- 不要把 `.env` 或任何 API Key commit 到 GitHub。
- 瀏覽器前端不能直接保存 provider API Key。
- 多人正式使用時，建議加登入，避免使用者猜測別人的 `thread_id`。
- 檔案型 JSONL memory 適合 MVP；正式多用戶版本建議改成 Postgres 或 SQLite。
