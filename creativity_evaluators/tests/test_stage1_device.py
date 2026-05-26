"""build_scorers must thread EvalConfig.stage1_device onto the Stage-1 model scorers, so a
GPU run can keep DSI/PPL/reward off the CPU. Construction is lazy (no weights loaded), so
this runs on a CPU-only box even with device='cuda'.
"""

from creativity_evaluators.pipeline import build_scorers
from creativity_evaluators.types import EvalConfig


def test_build_scorers_threads_stage1_device():
    s = build_scorers(EvalConfig(stage1_device="cuda", enable_stage2=False, run_reward=True))
    assert s.dsi.device == "cuda"
    assert s.ppl.device == "cuda"
    assert s.reward.device == "cuda"


def test_stage1_device_defaults_to_none():
    s = build_scorers(EvalConfig(enable_stage2=False, run_reward=True))
    assert s.dsi.device is None
    assert s.ppl.device is None
    assert s.reward.device is None
