# Conceptial — Live Demo Walkthrough

A 5-minute script for showing Conceptial cold to a creative-operations
audience. The flow is designed for in-person delivery on a laptop, but
works equally well as a recorded screencap.

## Pre-flight (do this once, before the day)

```bash
# 1. Seed the SQLite memory store with the three demo projects
uv run python scripts/seed_idea_memory.py --all --reset

# 2. Start the backend (port 8000)
uv run uvicorn brainstormlm.api.server:app --host 127.0.0.1 --port 8000

# 3. Start the frontend (port 5173) in a separate terminal
cd frontend && npm run dev
```

Open `http://localhost:5173/#/app` in a desktop browser, full-screen.

Confirm three projects appear on the dashboard with seeded counts:

- **Cypress & Salt — Brand Identity** (12 seeded ideas)
- **Bondi Trail — Q3 Campaign** (8 seeded ideas)
- **Larder — Oat Milk Packaging** (8 seeded ideas)

## The script

### Beat 1 — open a project, point at the memory chip (~30 sec)

> "This is Conceptial. It's a brainstorming engine built to return a set
> of distinct ideas rather than one polished answer."

Click into **Cypress & Salt — Brand Identity**.

Point at the memory chip in the bottom-right.

> "Every project has a private memory. This one already knows about 12
> visual-identity directions a designer has explored before. Watch what
> happens when we ask for more."

### Beat 2 — generate, watch ideas stream in (~60 sec)

Type into the seed input:

> *"Five fresh visual-identity directions that feel quietly expensive."*

Press enter. Five concept cards stream in over ~12 seconds.

The memory chip ticks up: `12 seeded · 1 explored`, then `12 seeded · 2`,
all the way to `12 seeded · 5 explored`.

> "None of those five overlap with the 12 already in the project's memory.
> Every prior direction, like 'Linen Folio', goes into an exclusion list,
> and candidates that match it are filtered out.
> Watch this."

### Beat 3 — regenerate, prove it doesn't repeat (~45 sec)

Click the seed node, hit the regenerate icon (or expand it again with a
slightly different sub-prompt).

> *"Show me five more — different from those five, different from the
> twelve in memory."*

A second wave of five concept cards streams in. The memory chip reaches
`12 seeded · 10 explored`.

> "Every new idea is checked against everything already in this
> project's memory, so the list keeps growing instead of repeating."

### Beat 4 — switch projects, prove memory is scoped (~45 sec)

Back to the dashboard. Open **Larder — Oat Milk Packaging**.

Memory chip reads: `8 seeded · 0 explored`.

> "Different project. Different memory. None of what we just generated
> for the skincare brand leaks here."

Type:

> *"Five packaging directions, refrigerated case, $7-9 carton."*

A fresh wave of five completely different ideas streams in — packaging-
specific, refrigerated-case-appropriate.

> "Each project has its own memory, stored in SQLite, so it is still
> there when you come back to the project."

### Beat 5 — the close (~30 sec)

> "Conceptial gives the team a wide set of starting points; the team
> picks the ones worth developing."

## Resilience

If the live backend has a bad moment during the demo, set
`CONCEPTIAL_DEMO_MODE=true` and the streaming endpoint serves
pre-cached JSON instead of running through the source. See
`docs/cloud_runpod_pivot.md` for the cache-builder script.

The frontend also runs end-to-end in pure mock mode by setting
`VITE_USE_MOCK_DATA=true` — the same project-scoped memory behaviour
works against localStorage instead of the SQLite store. Useful for
demos on a plane or any environment without backend.

## Where the content comes from

- **Seeded ideas**: hand-authored under `artifacts/conceptial/seeds/`.
  Loaded into SQLite via `scripts/seed_idea_memory.py`.
- **Generated ideas**: drawn from `src/brainstormlm/conceptial/demo_packs.py`,
  a 24-entry curated catalogue spanning three creative-ops axes
  (brand-identity, campaign-craft, object-design). The streaming source
  rotates through this catalogue deterministically and filters against
  the project's exclusion list, so re-running the demo on the same
  project always produces the same sequence.

## Reset between demos

```bash
# Wipe the SQLite store for a fresh start
rm -f data/idea_memory.db
uv run python scripts/seed_idea_memory.py --all

# Clear browser localStorage if in mock mode
# (DevTools → Application → Local Storage → clear all)
```

## Known caveats

- The streaming source uses a hand-authored catalogue
  (`demo_packs.py`), not a live model, on the demo path. Say so if asked.
- The demo shows text-level anti-mode + memory only; no activation
  steering runs on the demo path.
