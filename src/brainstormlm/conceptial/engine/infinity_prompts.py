"""
Load ideation prompts for Infinite-Chats (Hub) benchmarks with explicit provenance.

Hub-first: local files are used only when ``infinity_local_path`` or env
``INFINITY_LOCAL_PATH`` / legacy ``INFINITY_CHAT_PATH`` is set (no implicit ``data/`` scan).
If neither Hub nor local yields ideation rows, raises :class:`InfinityPromptLoadError` (no substitute prompts).

Requires optional deps (``pip install -e ".[dev]"``): ``datasets``, ``pandas``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InfinityPromptSource(str, Enum):
    """Serialized on CSV as ``infinity_prompt_source``."""

    HUB = "infinity_hub"
    LOCAL = "infinity_local"
    UNAVAILABLE = "infinity_unavailable"


class InfinityPromptLoadError(RuntimeError):
    """Cannot load real Infinite-Chats ideation prompts from the Hub or an explicit local file."""


@dataclass
class InfinityPromptLoadReport:
    """Auditable record of prompt loading (Hub + optional local)."""

    source: InfinityPromptSource
    prompts: list[str]
    infinity_ideation_labels: frozenset[str]
    infinity_hub_dataset_id: str | None = None
    infinity_hub_reachable: bool | None = None
    infinity_hub_probe_message: str = ""
    infinity_hub_train_row_count: int | None = None
    infinity_hub_ideation_pool_size: int | None = None
    infinity_local_candidates_checked: list[str] = field(default_factory=list)
    infinity_local_path_used: str | None = None
    infinity_sampled_prompt_count: int = 0
    infinity_requested_prompt_count: int = 0
    infinity_failure_notes: list[str] = field(default_factory=list)

    def to_dict(self, *, include_prompts: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["infinity_ideation_labels"] = sorted(self.infinity_ideation_labels)
        if not include_prompts:
            d["prompts"] = f"<{len(self.prompts)} prompts omitted>"
        return d


# --- Canonical constants (use these names in new code) ---

INFINITY_HUB_DATASET_ID = "liweijiang/infinite-chats-taxonomy"

INFINITY_IDEATION_LABELS = frozenset(
    {
        "brainstorming and ideation",
        "ideation and brainstorming",
    }
)

INFINITY_IDEATION_ALIAS = "brainstorming and ideation"

# Deprecated aliases (prefer INFINITY_* above)
IDEATION_LABELS_NORMALIZED = INFINITY_IDEATION_LABELS
DEFAULT_HUB_DATASET_ID = INFINITY_HUB_DATASET_ID
DEFAULT_CATEGORY_ALIAS = INFINITY_IDEATION_ALIAS


def _resolve_infinity_local_path_candidates(
    infinity_local_path: str | None,
) -> list[Path]:
    """Explicit paths only (Hub-first; no repo ``data/infinity_chat.*`` guessing)."""
    out: list[Path] = []
    if infinity_local_path:
        out.append(Path(infinity_local_path))
    env_primary = os.environ.get("INFINITY_LOCAL_PATH")
    if env_primary:
        out.append(Path(env_primary))
    legacy = os.environ.get("INFINITY_CHAT_PATH")
    if legacy:
        out.append(Path(legacy))
    return out


def probe_infinity_hub_dataset(infinity_hub_dataset_id: str) -> tuple[bool, str]:
    """Check Hub reachability (streaming first row). Returns (ok, message)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        return False, f"`datasets` not installed: {e}"

    try:
        ds = load_dataset(infinity_hub_dataset_id, split="train", streaming=True)
        first = next(iter(ds))
        keys = sorted(first.keys()) if isinstance(first, dict) else []
        return True, f"OK: streaming read succeeded; example keys: {keys}"
    except Exception as e:
        return False, f"FAILED: {type(e).__name__}: {e}"


def probe_hub_dataset(dataset_id: str) -> tuple[bool, str]:
    """Deprecated alias for :func:`probe_infinity_hub_dataset`."""
    return probe_infinity_hub_dataset(dataset_id)


def format_provenance_banner(report: InfinityPromptLoadReport) -> str:
    lines = [
        "=" * 78,
        "INFINITY BENCHMARK — PROMPT PROVENANCE (AUDIT)",
        "=" * 78,
        f"infinity_prompt_source: {report.source.value}",
        "",
        "Infinity Hub:",
        f"  infinity_hub_dataset_id:      {report.infinity_hub_dataset_id or '(none)'}",
        f"  infinity_hub_reachable:       {report.infinity_hub_reachable}",
        f"  infinity_hub_probe_message:   {report.infinity_hub_probe_message or '(not run)'}",
        f"  infinity_hub_train_row_count: {report.infinity_hub_train_row_count}",
        f"  infinity_hub_ideation_pool:   {report.infinity_hub_ideation_pool_size} rows (taxonomy filter)",
        "",
        "Infinity local (optional):",
        f"  infinity_local_candidates_checked: {report.infinity_local_candidates_checked or ['(none)']}",
        f"  infinity_local_path_used:          {report.infinity_local_path_used}",
        "",
        "Sampling:",
        f"  infinity_requested_prompt_count: {report.infinity_requested_prompt_count}",
        f"  infinity_sampled_prompt_count:   {report.infinity_sampled_prompt_count}",
        f"  infinity_ideation_labels:        {sorted(report.infinity_ideation_labels)}",
        "",
    ]
    if report.infinity_failure_notes:
        lines.append("infinity_failure_notes:")
        for n in report.infinity_failure_notes:
            lines.append(f"  - {n}")
        lines.append("")
    lines.append(
        "INTERPRETATION: infinity_hub = Hugging Face Infinite-Chats taxonomy (ideation slice); "
        "infinity_local = your file."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def _first_user_message_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def _example_has_ideation_category(example: dict) -> bool:
    for c in example.get("categories") or []:
        if isinstance(c, dict):
            name = str(c.get("category", "")).strip().lower()
            if name in INFINITY_IDEATION_LABELS:
                return True
    return False


def _find_category_column(df: Any) -> str | None:
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("category", "taxonomy", "categories", "topic", "type"):
            return c
    return None


def _find_prompt_column(df: Any) -> str | None:
    for name in ("prompt", "text", "instruction", "question", "input", "user", "content"):
        for c in df.columns:
            if str(c).lower().strip() == name:
                return c
    return None


def _path_to_dataframe(path: Path) -> Any | None:
    import pandas as pd
    from datasets import load_dataset

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            ds_dict = load_dataset("csv", data_files=str(path))
        elif suffix in (".json", ".jsonl"):
            ds_dict = load_dataset("json", data_files=str(path))
        else:
            return None
        split = next(iter(ds_dict.keys()))
        return ds_dict[split].to_pandas()
    except Exception:
        pass
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                for key in ("data", "rows", "items", "examples", "train", "test"):
                    if key in raw and isinstance(raw[key], list):
                        return pd.DataFrame(raw[key])
                return pd.DataFrame([raw])
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        pass
    return None


def load_ideation_prompts_for_benchmark(
    *,
    infinity_local_path: str | None,
    infinity_hub_dataset_id: str | None,
    use_infinity_hub: bool,
    infinity_prompt_count: int,
    rng: random.Random,
    infinity_ideation_alias: str = INFINITY_IDEATION_ALIAS,
) -> InfinityPromptLoadReport:
    """
    Resolve prompts: optional **explicit** local path first (ideation-filtered), else Hub.

    If neither yields ideation prompts, raises :class:`InfinityPromptLoadError`.

    Local is only considered for paths from ``infinity_local_path`` or env
    ``INFINITY_LOCAL_PATH`` / ``INFINITY_CHAT_PATH`` (legacy).
    """
    from datasets import load_dataset

    notes: list[str] = []
    local_checked: list[str] = []
    hub_id = infinity_hub_dataset_id or INFINITY_HUB_DATASET_ID

    hub_ok: bool | None = None
    hub_msg = ""
    if use_infinity_hub and hub_id:
        hub_ok, hub_msg = probe_infinity_hub_dataset(hub_id)
        if not hub_ok:
            notes.append(f"infinity_hub probe failed: {hub_msg}")

    search_paths = _resolve_infinity_local_path_candidates(infinity_local_path)

    for path in search_paths:
        local_checked.append(str(path))
        if not path.is_file():
            continue
        try:
            df = _path_to_dataframe(path)
            if df is None:
                notes.append(f"Could not parse as table: {path}")
                continue
            cat_col = _find_category_column(df)
            if cat_col is None:
                notes.append(f"No category/taxonomy column in {path}")
                continue
            prompt_col = _find_prompt_column(df)
            if prompt_col is None:
                obj_cols = [c for c in df.columns if df[c].dtype == object]
                prompt_col = obj_cols[0] if len(obj_cols) == 1 else None
            if prompt_col is None:
                notes.append(f"No prompt/text column in {path}")
                continue
            col_norm = df[cat_col].astype(str).str.strip().str.lower()
            allowed = INFINITY_IDEATION_LABELS | {infinity_ideation_alias.strip().lower()}
            mask = col_norm.isin(allowed)
            filtered = df.loc[mask, prompt_col].astype(str).str.strip()
            filtered = filtered[filtered.str.len() > 0].tolist()
            if not filtered:
                notes.append(f"No rows matched ideation filter in {path}")
                continue
            k = min(infinity_prompt_count, len(filtered))
            prompts = rng.sample(filtered, k=k)
            return InfinityPromptLoadReport(
                source=InfinityPromptSource.LOCAL,
                prompts=prompts,
                infinity_ideation_labels=INFINITY_IDEATION_LABELS,
                infinity_hub_dataset_id=hub_id,
                infinity_hub_reachable=hub_ok,
                infinity_hub_probe_message=hub_msg,
                infinity_local_candidates_checked=local_checked,
                infinity_local_path_used=str(path.resolve()),
                infinity_sampled_prompt_count=len(prompts),
                infinity_requested_prompt_count=infinity_prompt_count,
                infinity_failure_notes=notes,
            )
        except Exception as e:
            notes.append(f"Error reading {path}: {type(e).__name__}: {e}")
            continue

    if use_infinity_hub and hub_id:
        try:
            ds = load_dataset(hub_id, split="train")
            train_n = len(ds)
            filtered = ds.filter(_example_has_ideation_category)
            pool: list[str] = []
            for i in range(len(filtered)):
                p = _first_user_message_text(filtered[i]["messages"])
                if p:
                    pool.append(p)
            pool_n = len(pool)
            if pool:
                k = min(infinity_prompt_count, pool_n)
                prompts = rng.sample(pool, k=k)
                return InfinityPromptLoadReport(
                    source=InfinityPromptSource.HUB,
                    prompts=prompts,
                    infinity_ideation_labels=INFINITY_IDEATION_LABELS,
                    infinity_hub_dataset_id=hub_id,
                    infinity_hub_reachable=True,
                    infinity_hub_probe_message=hub_msg or "OK (full load succeeded)",
                    infinity_hub_train_row_count=train_n,
                    infinity_hub_ideation_pool_size=pool_n,
                    infinity_local_candidates_checked=local_checked,
                    infinity_sampled_prompt_count=len(prompts),
                    infinity_requested_prompt_count=infinity_prompt_count,
                    infinity_failure_notes=notes,
                )
            notes.append("Hub loaded but ideation pool empty after filter.")
        except Exception as e:
            notes.append(f"Hub full load failed: {type(e).__name__}: {e}")

    if not use_infinity_hub:
        notes.append("Hub disabled (--no-infinity-hub); no local ideation file succeeded.")
    else:
        notes.append(
            "Could not load ideation prompts from Infinite-Chats Hub or local file; "
            "benchmark cannot run without them."
        )

    msg_lines = [
        "Failed to load Infinite-Chats ideation prompts (Hub or explicit local).",
        f"hub_dataset_id={hub_id!r} use_infinity_hub={use_infinity_hub}",
    ]
    for n in notes:
        msg_lines.append(f"  - {n}")
    raise InfinityPromptLoadError("\n".join(msg_lines))


def print_provenance(report: InfinityPromptLoadReport, *, file=sys.stdout) -> None:
    print(format_provenance_banner(report), file=file, flush=True)


def format_connectivity_overview(
    *,
    infinity_local_path: str | None,
    infinity_hub_dataset_id: str,
    use_infinity_hub: bool,
) -> str:
    """Hub probe + existence of explicit local paths only."""
    lines = [
        "=" * 78,
        "INFINITY DATA — CONNECTIVITY (no model)",
        "=" * 78,
    ]
    hub_ok, hub_msg = (None, "(skipped)")
    if use_infinity_hub:
        hub_ok, hub_msg = probe_infinity_hub_dataset(infinity_hub_dataset_id)
    lines.append(f"infinity_hub_dataset_id: {infinity_hub_dataset_id}")
    lines.append(f"infinity_hub_reachable:  {hub_ok}")
    lines.append(f"infinity_hub_probe:      {hub_msg}")
    lines.append("")
    paths = _resolve_infinity_local_path_candidates(infinity_local_path)
    lines.append("infinity_local_path candidates (explicit env/CLI only):")
    if not paths:
        lines.append("  (none — Hub-only mode)")
    for p in paths:
        lines.append(f"  {'EXISTS' if p.is_file() else 'missing'}  {p}")
    lines.append("=" * 78)
    return "\n".join(lines)
