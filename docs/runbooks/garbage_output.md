# Runbook — model is outputting garbage

_Owner: whoever is on call. Phase 4 exit criterion 4._

## Symptoms

- Frontend shows incoherent or empty responses.
- `/healthz` returns 200 but `/ask` returns gibberish.
- Human feedback on `data/feedback_human.jsonl` skews to thumbs-down.

## Diagnose

1. **Read the latest server log lines.**
   ```sh
   tail -n 50 data/logs/server.jsonl | jq .
   ```
   Look for `"arm"` and `"adapter_path"` on recent `/ask` rows.

2. **Identify the active artifact.**
   ```sh
   jq . data/adapters/promoted.json
   ```
   If `adapter_path` points to a recently-flipped artifact, the candidate
   is suspect.

3. **Pin the regression to an artifact, not the prompt.**
   ```sh
   uv run brainstorm ask "name three lunch ideas" --no-adapter --no-cache
   ```
   This forces `TogetherGenerator` (the Phase-1 baseline). If output is
   sane → the local adapter or steering is the cause.

4. **Compare the new adapter against the prior on the held-out set.**
   ```sh
   uv run python scripts/eval_ablation.py --adapter <suspect> \
       --skip-memory
   ```
   The ablation table will show whether the regression is broad or
   prompt-specific.

## Mitigate

1. **Roll back immediately.** Edit `data/adapters/promoted.json` to point
   at the previous `run_id`:
   ```sh
   cp data/adapters/promoted.json{,.bak}
   # Find the previous good run_id from the JSONL backups or git log.
   jq '.adapter_path = "data/adapters/<previous_run_id>"
       | .run_id = "<previous_run_id>"' \
       data/adapters/promoted.json.bak > data/adapters/promoted.json
   ```
   The next `/ask` request reads the new pointer; no restart needed.

2. **Disable A/B routing of the bad arm.** If the regression is in arm
   `b`, set `traffic_a_pct: 100` in `data/ab_config.json` to drain traffic
   off `b`.

3. **If steering is involved**: blank `BRAINSTORMLM_STEERING_PATH`
   (or pass `--no-steered`) and re-test. Bad-strength steering vectors
   often produce coherent-but-off-topic text.

## Recover

1. File a follow-up: which gate did the bad adapter clear?
   - Inspect `data/eval/ablation_<run_id>_summary.json` for the run that
     promoted it.
   - If the held-out total looked good but live behavior is bad, the
     evaluator-vs-human agreement may be drifting → run
     `scripts/agreement.py` and tighten the rubric (`rubric_version` bump).
2. Tighten the promotion gate if needed (raise `min_improvement` or lower
   `max_p_value` in `training/promotion.py` defaults).

## Don't

- Don't truncate `data/feedback_human.jsonl` — it is append-only and
  feeds the next training run. Bad rows get filtered at mining time.
- Don't push to `main` with a half-promoted pointer. The atomic
  `os.replace` write means there is no half-promoted state on disk; if
  there is, something else is wrong.
