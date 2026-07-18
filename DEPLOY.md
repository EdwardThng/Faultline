# Deploying the FaultLine leaderboard

The leaderboard is a **static site**: `faultline report` writes `site/data/`
(a `leaderboard.json` plus one JSON per run under `traces/`), and
`site/index.html` reads those files directly and computes Wilson CIs in the
browser. There is no backend — any static host works.

## Automatic (GitHub Pages)

`.github/workflows/deploy.yml` builds and publishes on every push to `main`:

1. installs the package and runs the test suite,
2. runs the sweep with the **offline scripted agents** (no secrets needed),
3. generates `site/data/`,
4. uploads `site/` and deploys it to GitHub Pages.

One-time repo setup: **Settings → Pages → Build and deployment → Source =
GitHub Actions**. (Pages on a *private* repo requires a paid GitHub plan; make
the repo public — this benchmark is meant to be publicly inspectable anyway —
or host on Netlify/Cloudflare Pages instead.)

## Manual / local

```bash
pip install -e ".[dev]"
faultline sweep  --config sweeps/v1.yaml
faultline report --out site/data/
python -m http.server -d site 8000   # http://localhost:8000
```

Then publish the `site/` directory to any static host
(Netlify, Cloudflare Pages, `gh-pages`, S3, …).

## Publishing a real-model board

The default sweep uses the `scripted` / `scripted-stubborn` agents so the
pipeline is reproducible and offline. To benchmark real models:

1. Add provider credentials as repo secrets (e.g. `ANTHROPIC_API_KEY`).
2. In the workflow's **Run sweep** step, expose them as `env:`.
3. Point `--config` at a sweep whose `models:` list holds real model ids
   (e.g. `claude-sonnet-4-6`); the reference `ClaudeAgent` drives them.

Generated artifacts (`runs/*`, `site/data/*`) are gitignored — they are built
fresh on every deploy, never committed.
