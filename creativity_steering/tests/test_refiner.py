"""Concept-frozen OpenAI refiner — offline tests with a fake OpenAI client (no network).

The refiner polishes steered candidates for clarity/specificity but must stay concept-frozen
(no merging/reordering/homogenizing), which is enforced by its system prompt. We assert the call
shape + the guardrails are present, using an injected fake client so nothing hits the network.
"""

from creativity_steering.serving.refiner import REFINER_SYSTEM, OpenAIRefiner


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        return _FakeResp("REFINED OUTPUT")


class _FakeChat:
    def __init__(self, outer):
        self.completions = _FakeCompletions(outer)


class _FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = _FakeChat(self)


def test_refine_returns_model_content():
    fake = _FakeOpenAI()
    out = OpenAIRefiner(client=fake, model="gpt-4o-mini").refine("Prompt?", "1. idea A\n2. idea B")
    assert out == "REFINED OUTPUT"
    assert len(fake.calls) == 1


def test_refine_uses_concept_frozen_system_and_includes_inputs():
    fake = _FakeOpenAI()
    OpenAIRefiner(client=fake).refine("Design a logo.", "1. mascot owl\n2. wordmark")
    call = fake.calls[0]
    assert call["model"]  # a model was passed
    msgs = call["messages"]
    system = next(m["content"] for m in msgs if m["role"] == "system")
    user = next(m["content"] for m in msgs if m["role"] == "user")
    low = system.lower()
    # the load-bearing concept-frozen guardrails must be in the system prompt
    assert "concept" in low
    assert "merge" in low  # "do not merge ..."
    assert "reorder" in low or "same order" in low
    # the prompt + candidates both reach the model
    assert "Design a logo." in user and "mascot owl" in user and "wordmark" in user


def test_refine_respects_custom_model_and_temperature():
    fake = _FakeOpenAI()
    OpenAIRefiner(client=fake, model="gpt-4o", temperature=0.1).refine("p", "1. x")
    call = fake.calls[0]
    assert call["model"] == "gpt-4o" and call["temperature"] == 0.1


def test_refine_handles_none_content():
    fake = _FakeOpenAI()
    fake.chat.completions.create = lambda **kw: _FakeResp(None)  # API returned no content
    assert OpenAIRefiner(client=fake).refine("p", "1. x") == ""


def test_refiner_info_reports_model():
    assert OpenAIRefiner(client=_FakeOpenAI(), model="gpt-4o-mini").info()["model"] == "gpt-4o-mini"


def test_refiner_system_default_is_concept_frozen():
    low = REFINER_SYSTEM.lower()
    assert "concept" in low and "merge" in low and "distinct" in low
