# Scheduling the mart rebuild

The `reporting_state_health` mart is a **scheduled batch job**, exactly like a
production data pipeline — the Workbench UI is just a *consumer* that reads the
built artifacts and shows their freshness. Nothing in the app rebuilds the mart
on page load; a scheduler (or you) runs the builder, and the mart screen's
"Last built" badge tells everyone how fresh the data is.

## What the rebuild does

```
venv\Scripts\python.exe scripts\build_mart.py --refresh
```

`--refresh` re-pulls CDC / CMS / Census from source (ignoring the local cache),
re-assembles the 51-state mart, writes `outputs/mart/reporting_state_health.parquet`
and `outputs/mart/build_meta.json` (with `built_at`, per-source row counts,
sample-mode flag, and the measures included), and loads the DuckDB table.

Useful flags:

- `--full-cms` — pull all ~1.3M CMS providers instead of the 300k sample.
- `--measures DIA01 NPW14 TOB04 ...` — choose which CDI measures to include.

## Weekly rebuild — Windows Task Scheduler (one-liner)

Run this once in an elevated `cmd` / PowerShell, editing the two absolute paths
to match your checkout. It creates a task that rebuilds every Sunday at 02:00:

```bat
schtasks /Create /SC WEEKLY /D SUN /ST 02:00 /TN "HealthPipe Mart Rebuild" /TR "\"C:\Users\<you>\Python_Projects\healthpipe_ai_v2\venv\Scripts\python.exe\" \"C:\Users\<you>\Python_Projects\healthpipe_ai_v2\scripts\build_mart.py\" --refresh"
```

Manage it later:

```bat
schtasks /Run    /TN "HealthPipe Mart Rebuild"   :: run now
schtasks /Query  /TN "HealthPipe Mart Rebuild"   :: check status / last run
schtasks /Delete /TN "HealthPipe Mart Rebuild" /F
```

## Freshness on screen and in preflight

- **Mart screen** shows "Last built: N day(s) ago" and a **STALE > 7 DAYS** warn
  badge (plus a *Refresh now* button) when the mart is older than a week.
- **`python scripts/preflight.py`** includes a "Mart freshness" row — ✅ when
  fresh, ⚠️ when older than 7 days or never built — so the demo-readiness check
  surfaces staleness before you present.
