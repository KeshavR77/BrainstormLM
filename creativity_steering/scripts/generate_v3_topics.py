"""Generate ~1100 diverse brainstorming topics via the Anthropic API.

Used to grow the topic pool for v3 derive (target: 1000 prompts → 100 each at top/bottom
10%). Writes ``creativity_steering/data/v3_topics.json`` in the load_prompt_set format.

The Anthropic API is called in batches (50 topics per call) with a strong dedup pass at the
end. Reads ANTHROPIC_API_KEY from .env at the repo root.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "creativity_steering" / "data"


def load_anthropic_key() -> str:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not in .env or environment")
    return key


GEN_PROMPT = (
    "Generate {n} diverse, concise creative-brainstorming prompts in the same style as "
    "these examples (one prompt per line, no numbering, no bullet points, no commentary):\n\n"
    "{examples}\n\n"
    "Constraints:\n"
    "- Each prompt is one short sentence ending in a period.\n"
    "- Mix of verbs (Pitch / Design / Invent / Describe / Propose / Name / Reinvent / "
    "Concept / Suggest / Reimagine).\n"
    "- Mix of domains (everyday objects, food, urban life, art, science, nature, work, "
    "play, future tech, social rituals).\n"
    "- Avoid repeating ideas already in the examples or earlier in your output.\n"
    "- Do NOT generate prompts that ask for code, software, apps, or technical implementations.\n"
    "- Stay open-ended and inviting — these are creative warm-ups, not constraint puzzles.\n\n"
    "Output exactly {n} prompts."
)


SEED_EXAMPLES = (
    "Pitch a new consumer coffee brand.",
    "Design a playground for an elderly community.",
    "Invent a holiday for neighbours.",
    "Describe a garden that works at night.",
    "Propose a slogan for a bicycle co-op.",
    "Name a constellation that does not exist yet.",
    "Reinvent the umbrella.",
    "Concept a community seed library.",
    "Suggest a ritual for finishing a big project.",
    "Reimagine the morning alarm clock.",
)


def call_claude(client, n: int, seen_lines: list[str]) -> list[str]:
    """One generation batch. ``seen_lines`` is sampled for examples + nudges to avoid repeats."""
    import random

    examples = SEED_EXAMPLES + tuple(random.sample(seen_lines, min(15, len(seen_lines))))
    examples_block = "\n".join(examples)
    prompt = GEN_PROMPT.format(n=n, examples=examples_block)

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    # Strip leading numbering / bullets if the model defied the instruction.
    cleaned = []
    for line in raw:
        line = re.sub(r"^\s*[\-\*\d\.\)\(]+[\s\.\)\:\-]*", "", line).strip()
        if not line:
            continue
        if not line.endswith("."):
            line = line.rstrip(".!?,;") + "."
        cleaned.append(line)
    return cleaned


def main(target: int = 1100, batch: int = 50, out_path: Path | None = None) -> None:
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    key = load_anthropic_key()
    client = anthropic.Anthropic(api_key=key)

    out_path = out_path or (DATA_DIR / "v3_topics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topics: list[str] = []
    seen: set[str] = set()

    def normalize(t: str) -> str:
        return re.sub(r"\s+", " ", t.lower().strip().rstrip("."))

    # Grow an existing bank in place: keep prior topics AND their order, so previously generated
    # responses + cached judgments stay valid (seeds are positional) and only NEW unique topics
    # are appended up to ``target``.
    if out_path.exists():
        prior = json.loads(out_path.read_text()).get("topics", [])
        for t in prior:
            k = normalize(t)
            if k not in seen:
                seen.add(k)
                topics.append(t)
        print(f"[gen] loaded {len(topics)} existing topics from {out_path} (append mode)")

    attempts = 0
    while len(topics) < target and attempts < 60:
        attempts += 1
        n = min(batch, target - len(topics) + 20)
        try:
            batch_topics = call_claude(client, n, topics)
        except Exception as e:
            print(f"[gen] batch {attempts} failed: {e}; retrying after backoff", file=sys.stderr)
            time.sleep(5 * attempts)
            continue

        new_count = 0
        for t in batch_topics:
            key_t = normalize(t)
            if key_t in seen or len(key_t) < 8:
                continue
            seen.add(key_t)
            topics.append(t)
            new_count += 1
        print(f"[gen] batch {attempts}: +{new_count} new (pool={len(topics)})")
        if new_count < 5:
            # The model is starting to dup heavily; bigger nudge next round.
            pass

    topics = topics[:target]
    payload = {
        "creative_system": "You are a wildly inventive brainstorming partner. Embrace "
        "surprising ideas, unconventional analogies, and creative leaps. Be original. "
        "Quality over consensus.",
        "neutral_system": "You are a careful, conventional assistant. Give the most "
        "standard, expected, straightforward answer. Avoid speculation or unusual ideas.",
        "topics": topics,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[gen] wrote {len(topics)} topics -> {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=1100)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    main(target=args.target, batch=args.batch, out_path=args.out)
