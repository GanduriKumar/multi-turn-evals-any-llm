# Scripts (what they do)

These are small helpers to run and stop the app easily on Windows.

Most useful
- dev.ps1 — runs backend and frontend for local development
- start-detached.ps1 — runs them in separate windows that keep running
- stop.ps1 — stops the default dev ports (8000 and 5173)
- smoke.ps1 — quick check that the backend is alive and has datasets

Examples
Run both servers now:
```powershell
./scripts/dev.ps1
```

Keep them running even if you close VS Code:
```powershell
./scripts/start-detached.ps1
```

Stop them:
```powershell
./scripts/stop.ps1
```
