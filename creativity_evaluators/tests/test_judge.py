"""Tests for the judge wrapper logic (parsing, clamping, truncation, retry, caching).

The provider SDK calls themselves are exempt; we drive a ``ScriptedJudge`` whose ``_call``
returns canned strings (or raises), so all assertions run offline.
"""

from __future__ import annotations

import json

import pytest

from creativity_evaluators.stage2.judge import BaseJudge, parse_judge_json
from creativity_evaluators.types import DIMENSIONS, JudgeResult
from creativity_evaluators.utils.caching import DiskCache

VALID = json.dumps({**{d: 4 for d in DIMENSIONS}, "reasoning": "ok"})


class ScriptedJudge(BaseJudge):
    """A judge whose ``_call`` replays a scripted list of returns/exceptions."""

    def __init__(self, script, **kw):
        super().__init__(name="scripted", model="m", family="test", backoff_base=0.0, **kw)
        self._script = list(script)
        self.calls = 0
        self.last_user: str | None = None

    def _call(self, system: str, user: str) -> str:
        self.last_user = user
        self.calls += 1
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def test_parse_valid_json():
    res = parse_judge_json(VALID, "j")
    assert isinstance(res, JudgeResult)
    assert res.scores["Originality"] == 4
    assert res.judge_name == "j"


def test_parse_clamps_and_rounds_out_of_range():
    raw = json.dumps({**{d: 3 for d in DIMENSIONS}, "Originality": 9, "Fluency": 0})
    res = parse_judge_json(raw, "j")
    assert res.scores["Originality"] == 5
    assert res.scores["Fluency"] == 1


def test_parse_strips_code_fences():
    res = parse_judge_json("```json\n" + VALID + "\n```", "j")
    assert res.scores["Pragmaticality"] == 4


def test_parse_missing_dimension_raises():
    bad = json.dumps({"Originality": 3, "reasoning": "x"})
    with pytest.raises(ValueError):
        parse_judge_json(bad, "j")


def test_score_returns_judgeresult():
    res = ScriptedJudge([VALID]).score("p", "r")
    assert res.judge_name == "scripted"


def test_response_truncated_before_judging():
    j = ScriptedJudge([VALID], truncate_chars=10)
    j.score("p", "x" * 5000)
    assert j.last_user is not None
    assert j.last_user.count("x") <= 10


def test_retries_then_succeeds():
    j = ScriptedJudge([RuntimeError("boom"), RuntimeError("boom"), VALID])
    res = j.score("p", "r")
    assert res.scores["Originality"] == 4
    assert j.calls == 3


def test_retry_exhausted_raises():
    j = ScriptedJudge([RuntimeError("boom")], max_retries=2)
    with pytest.raises(RuntimeError):
        j.score("p", "r")
    assert j.calls == 2


def test_cache_hit_avoids_second_call(tmp_path):
    j = ScriptedJudge([VALID], cache=DiskCache(tmp_path))
    a = j.score("p", "r")
    b = j.score("p", "r")
    assert a.scores == b.scores
    assert j.calls == 1
