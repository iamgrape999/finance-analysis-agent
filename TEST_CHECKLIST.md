# CathyChang AI Deployment Test Checklist

Use this checklist before and after each deployment.

Goals:

- verify multi-user isolation
- verify request-scoped provider/model overrides
- verify provider probe does not pollute chat requests
- verify web search and dark mode still work

Project path:

`C:\Users\HAN-LI CHANG\Documents\Codex\2026-04-19-colab-python-3-github-ai-writefile`

---

## 0. Preconditions

- Python is installed:
  - `py --version`
- Required `.env` or Render environment variables are configured
- At least one provider key is valid
- `APP_PASSWORD` or `APP_USERS` is configured
- `X-User-Id` isolation is enabled in the backend

---

## 1. Local compile check

Run:

```powershell
cd "C:\Users\HAN-LI CHANG\Documents\Codex\2026-04-19-colab-python-3-github-ai-writefile"
py -m py_compile app.py Finance_Analysis_Agent_V581.py
```

Expected:

- no output
- no `SyntaxError`
- no `IndentationError`

---

## 2. Health check

Open:

- `https://finance-analysis-agent.onrender.com/api/healthz`

Expected:

- valid JSON response
- not a 500 error
- fields such as:
  - `backend_version`
  - `nvidia_key_present`
  - `mistral_import_ok`
  - `web_search_enabled`

Check:

- provider key flags match the actual environment
- model fields look correct
- `*_effective_model` matches expectation when aliases are used

---

## 3. Single-user override test

Purpose:

- verify one user can switch provider/model across requests
- verify the old request does not leak into the next request

### Test A: Mistral only

1. Log in as user `hanli`
2. In the UI set:
   - `Provider Order` = `mistral only (no fallback)`
   - `Mistral Model` = `mistral-small-latest`
3. Send:

```text
hi
```

Expected:

- reply card shows:
  - `platform=mistral`
  - `model=mistral-small-latest`

### Test B: Switch to Cerebras only

1. Stay on user `hanli`
2. In the UI set:
   - `Provider Order` = `cerebras only (no fallback)`
   - `Cerebras Model` = `llama3.1-8b`
3. Send:

```text
hi
```

Expected:

- reply card now shows:
  - `platform=cerebras`
  - `model=llama3.1-8b`
- it must not still use the previous Mistral setting

---

## 4. Multi-user isolation test

Purpose:

- verify user A and user B do not affect each other

Setup:

- use two browsers, or
- one normal window and one private/incognito window

### Window A

- user id: `hanli`
- provider order: `mistral only (no fallback)`
- model: `mistral-small-latest`
- message:

```text
hi from hanli
```

### Window B

- user id: `cathy`
- provider order: `cerebras only (no fallback)`
- model: `llama3.1-8b`
- message:

```text
hi from cathy
```

Expected:

- A uses `mistral`
- B uses `cerebras`
- no cross-contamination
- thread history remains separated by user

---

## 5. Provider probe pollution test

Purpose:

- verify `/api/provider-probe/...` does not change later chat behavior

Steps:

1. In chat UI set:
   - `Provider Order` = `mistral only (no fallback)`
   - `Mistral Model` = `mistral-small-latest`
2. Send:

```text
hi
```

3. Run one `nvidia` probe
4. Return to chat UI and send:

```text
hi again
```

Expected:

- first chat uses `mistral`
- probe may succeed or fail
- second chat must still use `mistral`
- the probe must not switch the later chat to `nvidia`

---

## 6. Log Panel verification

Purpose:

- verify request-scoped override behavior from logs

Expected patterns:

```text
[CALL] calling /api/chat; user=hanli; thread=...
[OK] reply ok; provider=mistral model=mistral-small-latest
```

Check:

- `user=...` is correct
- `provider_order=...` matches the UI choice
- `provider=` and `model=` match the current request

Must not happen:

- user A selects `mistral only` but log shows `nvidia`
- after probe, chat provider changes unexpectedly
- two users see each other's provider/model behavior

---

## 7. Web search verification

Purpose:

- verify intent detection, source display, and Tavily/Serper switching

### Test A: Auto search

1. Set `Web Search Source` = `Auto`
2. Send:

```text
What are the latest NVIDIA NIM models?
```

Expected:

- reply card shows a search badge:
  - `Web search: tavily`
  - or `Web search: serper`
- source list appears below the answer
- Log Panel shows:
  - `web search decision: ...`
  - `web search provider: ...`

### Test B: Tavily only

1. Set `Web Search Source` = `Tavily`
2. Enable `Force Web Search`
3. Send:

```text
What did the Taiwan stock market close at today?
```

Expected:

- card badge shows `Web search: tavily`
- Log Panel shows:
  - `web search request: provider=tavily; force=true`
  - `web search provider: tavily`

### Test C: Serper only

1. Set `Web Search Source` = `Serper`
2. Enable `Force Web Search`
3. Send:

```text
What models are currently online in Cerebras?
```

Expected:

- card badge shows `Web search: serper`
- Log Panel shows:
  - `web search request: provider=serper; force=true`
  - `web search provider: serper`

---

## 8. Dark mode verification

Purpose:

- verify dark mode toggle works and persists

Steps:

1. Click the top-right theme toggle
2. Confirm these areas switch correctly:
   - topbar
   - sidebar
   - settings panel
   - log panel
   - chat cards
3. Refresh the page

Expected:

- dark mode remains active after refresh
- clicking again switches back to light mode

---

## 9. Minimum deployment acceptance

Treat the deployment as accepted if all of these pass:

1. local compile check passes
2. `healthz` returns valid JSON
3. multi-user chat does not cross-contaminate
4. provider probe does not pollute chat flow
5. web search and dark mode both work

---

## 10. Fast smoke test before deployment

If time is short, run at least this:

1. `py -m py_compile app.py Finance_Analysis_Agent_V581.py`
2. open `healthz`
3. user `hanli` with `mistral only` sends `hi`
4. user `cathy` with `cerebras only` sends `hi`
5. run one `nvidia` probe
6. go back to `hanli` and send `hi`
7. test one search query:
   - `What are the latest NVIDIA NIM models?`

If there is no cross-user contamination, no probe pollution, and no 500 error, the build is usually safe to deploy.
