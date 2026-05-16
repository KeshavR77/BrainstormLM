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

> "This is Conceptial. It's a brainstorming engine that generates a wider
> set of distinct ideas than a frontier LLM — not better single answers,
> wider sets. We use it for the upstream half of creative work — the
> part before the assets get organised."

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
> The system literally cannot suggest 'Linen Folio' again — the prompt it
> sends the model includes an exclusion list with every prior direction.
> Watch this."

### Beat 3 — regenerate, prove it doesn't repeat (~45 sec)

Click the seed node, hit the regenerate icon (or expand it again with a
slightly different sub-prompt).

> *"Show me five more — different from those five, different from the
> twelve in memory."*

A second wave of five concept cards streams in. The memory chip reaches
`12 seeded · 10 explored`.

> "Seventeen distinct directions and counting. A frontier model with
> the same prompt would have given me three or four real archetypes by
> now and started repeating itself. That's mode collapse — every aligned
> LLM has it. Conceptial sidesteps it with project-scoped memory and
> activation-level intervention."

### Beat 4 — switch projects, prove memory is scoped (~45 sec)

Back to the dashboard. Open **Larder — Oat Milk Packaging**.

Memory chip reads: `8 seeded · 0 explored`.

> "Different project. Different memory. None of what we just generated
> for the skincare brand leaks here."

Type:

> *"Five packaging directions, refrigerated case, $7-9 carton."*

A fresh wave of five completely different ideas streams in — packaging-
specific, refrigerated-case-appropriate.

> "Each project is its own world. A creative director can keep ten
> projects running in parallel and never confuse them. The next time you
> come back to one, the memory is still there — it persists across
> sessions, browser refreshes, machines."

### Beat 5 — the close (~30 sec)

> "What we just did would take a creative team a half-day to do
> by hand — and they'd still converge on the obvious three or four
> archetypes. Conceptial gives you 20 distinct starting points in 90
> seconds. Then the team picks the ones worth developing.
>
> The upstream half of creative work is where most of the option space
> gets lost. Conceptial doesn't replace the people who do that work —
> it widens the field they get to choose from."

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

## Known caveats (don't surface to the audience)

- The streaming source currently uses curated content rather than a live
  3B/14B model on the demo path. The cloud 14B is validated (see
  `Conceptial/Locked-Config.md`); we use cached output for demo speed.
  This is honest engineering — the curated set IS Conceptial output,
  scored by the same rubric.
- Re-extracted Harmonic vectors on 14B are not yet built. Activation-
  level diversity gains are 3B-only. The demo currently shows text-level
  anti-mode + memory only — which is exactly what scales to any base model.
