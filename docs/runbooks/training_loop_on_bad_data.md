# Runbook — training loop is looping on bad data

_Owner: whoever is on call. Phase 4 exit criterion 4._

## Symptoms

- `scripts/continuous_train.py` runs but `promoted.json` never flips.
- Sequential runs show "improvement below threshold" or "p-value too high"
  for many cycles in a row.
- `train_metrics.csv` shows DPO loss flat or NaN.
- Mined preference pairs collapsed to a few responses each (low diversity).

## Diagnose

1. **Inspect the latest pair-mining outputs.**
   ```sh
   wc -l data/preferences.jsonl
   jq '.score_gap' data/preferences.jsonl | sort -n | uniq -c | tail
   ```
   Median `score_gap` < 2.5 → evaluator is too noisy or rubric is drifting.

2. **Check evaluator-vs-human agreement.**
   ```sh
   uv run python scripts/agreement.py --out data/eval/agreement_now.json
   jq '.signed_agreement, .pearson' data/eval/agreement_now.json
   ```
   If `signed_agreement < 0.6` or `pearson < 0.3`, the synthetic signal is
   misaligned with humans — training on it will not help live users.

3. **Inspect human feedback distribution.**
   ```sh
   jq -c 'select(.human_rating != null) | .human_rating.thumbs' \
       data/feedback_human.jsonl | sort | uniq -c
   ```
   If thumbs-down ≫ thumbs-up, the deployed model is bad in the wild;
   training on those pairs amplifies whatever "down" responses look like.

4. **Check for caching collisions.**
   ```sh
   jq -c '[.prompt, .response] | @csv' data/preferences.jsonl | sort | uniq -d
   ```
   Duplicate (prompt, response) tuples on opposite sides of pairs → cache
   key collision; bump generator temperature schedule in
   `scripts/sample_siblings.py`.

## Mitigate

1. **Stop the loop.** If a Modal cron is wired, pause it. If the operator
   is the trigger, just stop running the script.
2. **Quarantine the suspect rows.** Move a window of human feedback
   aside:
   ```sh
   mv data/feedback_human.jsonl data/feedback_human.jsonl.quarantine
   touch data/feedback_human.jsonl
   ```
   Rerun `continuous_train.py --skip-mining` over the surviving
   `data/preferences.jsonl` to confirm the pipeline is healthy without
   the new feedback.
3. **Bump `rubric_version`** if the evaluator is the issue; the new
   version will not retroactively recompute old scores, but new scoring
   runs use the updated rubric.
4. **Roll back to the last known-good adapter** by editing
   `data/adapters/promoted.json` (see `garbage_output.md`).

## Recover

1. Re-introduce the quarantined feedback in chunks; rerun the
   continuous-train pipeline after each batch and watch the gate
   decisions. The first chunk that produces a "passed gate" outcome is
   your safe re-entry point.
2. If the issue was rubric drift, schedule a calibration pass: pick 30
   live trajectories, score by hand, update the evaluator system prompt,
   bump `rubric_version`.

## Don't

- Don't train on the held-out 50 prompts — `eval/fixtures.HELD_OUT` is
  asserted-disjoint at mining time. If you suspect a leak, grep
  preferences.jsonl for those prompt strings.
- Don't reuse a `run_id` — promoted-pointer history is keyed on it.
- Don't bypass the promotion gate. The whole point is that a bad run
  cannot reach live traffic.
