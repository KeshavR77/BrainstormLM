"""
llm_service.py — All OpenAI communication for the Recursive What-If
Cascade (WIC) Engine.

Four core functions (extractor, inversion generator, simulator,
synthesizer) plus a safe_llm_call retry wrapper that forces JSON output.
"""

import json
import os
from collections.abc import AsyncIterator
from json import JSONDecodeError
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES = 2


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════

EXTRACTOR_SYSTEM = """
You are an assumption archaeologist. Your only job is to find the single 
most load-bearing, invisible assumption inside a problem statement or 
simulated reality.

DEFINITIONS:
- Load-bearing: If this assumption is false, the problem as stated 
  cannot be solved in its current form.
- Invisible: The person who stated the problem almost certainly did not 
  consciously notice this assumption. Do NOT surface obvious constraints 
  they stated explicitly.

RULES:
1. Output exactly ONE assumption. Not two. Not a list.
2. The assumption must be falsifiable — there must exist a real or 
   hypothetical world where it is not true.
3. Do NOT surface assumptions about resources, time, or money unless 
   they are the ONLY load-bearing assumption. Prefer assumptions about 
   causality, definitions, substrate, or actor identity.
4. Your rationale must explain WHY this assumption is invisible (why 
   most people would not question it).

Prior inversions already explored in this session (do NOT re-surface 
these or closely related assumptions):
{prior_inversions_list}

OUTPUT FORMAT (JSON only, no prose, no markdown):
{
  "assumption": "<single declarative sentence stating what is assumed true>",
  "rationale": "<2 sentences: why it's load-bearing, why it's invisible>",
  "assumption_type": "<one of: causal | definitional | scoping | metric | actor | substrate>"
}
"""

INVERSION_SYSTEM: dict[str, str] = {
    "opposite": """
You generate the OPPOSITE inversion of an assumption.
The opposite inversion takes the assumption and asserts its logical negation.
If the assumption is "X causes Y", the opposite is "Y causes X" or "X does not cause Y".
If the assumption is "users want Z", the opposite is "users actively avoid Z".

The inversion must be:
- Logically coherent (not self-contradictory)
- Non-trivial (not just adding "not" to the sentence)
- Specific enough to constrain a solution

OUTPUT FORMAT (JSON only):
{
  "inversion_statement": "<What if [inverted reality]? — full sentence>",
  "one_line_implication": "<The immediate first-order consequence of this being true>"
}
""",
    "extremity": """
You generate the EXTREMITY inversion of an assumption.
The extremity inversion takes the assumption and pushes it to its logical maximum 
or minimum — to the point where it breaks or transforms into something qualitatively 
different.
If the assumption is "users pay for value", extremity is "users pay for everything, 
including breathing air" OR "users pay absolutely nothing and are paid to participate".

The inversion must:
- Be at a true extreme, not just a large increment
- Create a qualitatively different world, not just a quantitative shift
- Remain physically possible (not require magic)

OUTPUT FORMAT (JSON only):
{
  "inversion_statement": "<What if [extreme reality]? — full sentence>",
  "one_line_implication": "<The immediate first-order consequence of this being true>"
}
""",
    "alien": """
You generate the ALIEN SUBSTRATE inversion of an assumption.
The alien substrate inversion replaces the medium, material, or system through 
which the assumption operates with something from a completely different domain.
If the assumption operates in software, move it to biology, physics, economics, 
or social systems. If it operates in physical space, move it to information space.

The inversion must:
- Import a mechanism from a domain at least 2 conceptual steps removed
- Not be a metaphor — it must be a genuine structural replacement
- Produce a world that is strange but internally consistent

OUTPUT FORMAT (JSON only):
{
  "inversion_statement": "<What if [alien substrate reality]? — full sentence>",
  "one_line_implication": "<The immediate first-order consequence of this being true>"
}
""",
}

SIMULATOR_SYSTEM = """
You are a world-simulator. You receive a reality inversion — a "What if?" statement 
that posits an alternative world — and you simulate its first and second-order 
consequences with logical rigor.

You are NOT brainstorming solutions. You are mapping the PHYSICS of the new world.
You are describing what is true in this world, not what someone should build.

RULES:
1. Every item in "breaks" must be something that concretely stops working 
   because of the inversion. No vague statements like "business models change."
   Say exactly which business model and exactly why it breaks.
2. Every item in "abundances" must be something that becomes cheaper, easier, 
   or more available specifically because of this inversion. 
3. "new_constraints" are things that LIMIT what is possible in this world that 
   were not constraints before. These are the raw material for the next assumption extraction.
4. "new_actors" are stakeholders who gain power, relevance, or existence in this world.
5. The simulation_narrative synthesizes the world in 2 sentences. 
   It should feel like a description from inside the world, not outside it.

Do not hedge. Do not say "might" or "could". Simulate as if this world is real.

OUTPUT FORMAT (JSON only, no prose, no markdown):
{
  "breaks": ["<specific thing 1>", "<specific thing 2>", "<specific thing 3>"],
  "abundances": ["<specific thing 1>", "<specific thing 2>", "<specific thing 3>"],
  "new_constraints": ["<constraint 1>", "<constraint 2>"],
  "new_actors": ["<actor 1>", "<actor 2>"],
  "simulation_narrative": "<2 sentences describing the texture of this world>"
}
"""

SYNTHESIZER_SYSTEM = """
You are an invention harvester. You receive the full trace of a recursive 
assumption-breaking cascade — a chain of "What if?" inversions and their 
simulated consequences — and you synthesize a single concrete product or 
service idea from the final simulated world.

THE IDEA MUST:
1. Be native to the final simulated world — it must REQUIRE the cascade's 
   inversions to be true to make sense. An idea that would work in the 
   original world without any inversions is a failure.
2. Exploit at least one "abundance" from the simulation as its core mechanism.
3. Be constrained by at least one "new_constraint" from the simulation.
4. Name a specific target user (not "businesses" or "users" — a real person 
   in a real role facing a real moment).
5. Include an MVT (minimum viable test): the cheapest experiment that would 
   tell you whether the core assumption of the idea is true. No MVT should 
   require more than $500 or 2 weeks to run.

The "assumption_chain" field must trace every assumption → inversion decision 
made in the cascade, in order from root to leaf. This is the idea's 
intellectual lineage.

OUTPUT FORMAT (JSON only, no prose, no markdown):
{
  "title": "<5 words max, evocative not descriptive>",
  "mechanism": "<How it works in 2-3 sentences. Mechanistic, not marketing.>",
  "target_user": "<Specific person: role, context, moment of need>",
  "core_value_prop": "<One sentence. Finish: 'This exists because in this world, ___'>",
  "assumption_chain": [
    "<Depth 0: assumed X, inverted to Y>",
    "<Depth 1: assumed A, inverted to B>",
    "<Depth 2: assumed P, inverted to Q>"
  ],
  "mvt": "<Specific experiment: what you do, what you measure, what result validates>",
  "adjacent_analogues": [
    "<Real thing that works similarly in a different domain>",
    "<Real thing that works similarly in a different domain>"
  ]
}
"""


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════


def build_extractor_prompt(
    problem_context: str,
    prior_inversions: list[str],
    depth: int,
) -> str:
    prior_str = (
        "Prior inversions explored:\n"
        + "\n".join(f"- {inv}" for inv in prior_inversions)
        if prior_inversions
        else ""
    )
    label = "ORIGINAL PROBLEM:" if depth == 0 else "SIMULATED REALITY TO ANALYZE:"
    return f"""
Depth in cascade: {depth} (0 = original problem, 1+ = simulated reality)

{label}
{problem_context}

{prior_str}

Extract the single most load-bearing invisible assumption.
"""


def build_inversion_prompt(
    assumption: str,
    problem_context: str,
    inversion_type_value: str,
) -> str:
    return f"""
Problem context: {problem_context}

Assumption to invert: "{assumption}"

Generate the {inversion_type_value} inversion of this assumption.
"""


def build_simulator_prompt(
    inversion_statement: str,
    implication: str,
    full_context: str,
) -> str:
    return f"""
Full cascade context so far:
{full_context}

Inversion to simulate: {inversion_statement}

First-order implication already identified: {implication}

Now simulate the full consequences of this world being real.
"""


# ═══════════════════════════════════════════════════════════════════════
# SAFE LLM CALL WRAPPER  (non-streaming)
# ═══════════════════════════════════════════════════════════════════════


async def safe_llm_call(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """
    Call the OpenAI Chat API with forced JSON output.

    On JSONDecodeError the error is appended to the user prompt and the
    call is retried up to ``retries`` times.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for attempt in range(1 + retries):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""

        try:
            return json.loads(raw)
        except JSONDecodeError as exc:
            last_error = exc
            # Append corrective feedback and retry
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was not valid JSON. "
                        f"Error: {exc}. "
                        f"Please respond with valid JSON only."
                    ),
                },
            )

    raise ValueError(
        f"Failed to get valid JSON after {1 + retries} attempts. "
        f"Last error: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════════
# CORE LLM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


async def call_extractor(
    problem_context: str,
    prior_inversions: list[str],
    depth: int,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Extract the single most load-bearing invisible assumption from a
    problem statement or simulated reality.

    Returns dict with keys: assumption, rationale, assumption_type
    """
    prior_str = (
        "\n".join(f"- {inv}" for inv in prior_inversions)
        if prior_inversions
        else "None yet."
    )
    system = EXTRACTOR_SYSTEM.replace("{prior_inversions_list}", prior_str)
    user = build_extractor_prompt(problem_context, prior_inversions, depth)

    return await safe_llm_call(system, user, model=model)


async def call_inversion_generator(
    assumption: str,
    problem_context: str,
    inversion_type_value: str,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Generate a single inversion branch (opposite / extremity / alien)
    for the given assumption.

    Returns dict with keys: inversion_statement, one_line_implication
    """
    system = INVERSION_SYSTEM[inversion_type_value]
    user = build_inversion_prompt(assumption, problem_context, inversion_type_value)

    return await safe_llm_call(system, user, model=model, temperature=0.9)


async def call_simulator(
    inversion_statement: str,
    implication: str,
    full_context: str,
    *,
    model: str = DEFAULT_MODEL,
) -> AsyncIterator[str]:
    """
    Simulate the consequences of an inversion.

    **Streams** text chunks as they arrive.  The caller should
    accumulate chunks and parse the final concatenated string as JSON.
    """
    user = build_simulator_prompt(inversion_statement, implication, full_context)

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SIMULATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def call_simulator_safe(
    inversion_statement: str,
    implication: str,
    full_context: str,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Non-streaming convenience wrapper around :func:`call_simulator`.

    Accumulates the full streamed response, parses it as JSON, and
    applies the same retry logic as ``safe_llm_call`` on parse failure.
    """
    last_error: Exception | None = None

    for attempt in range(1 + MAX_RETRIES):
        accumulated = ""
        async for chunk in call_simulator(
            inversion_statement, implication, full_context, model=model
        ):
            accumulated += chunk

        try:
            return json.loads(accumulated)
        except JSONDecodeError as exc:
            last_error = exc
            # The streaming API can't append corrective messages, so we
            # simply retry the entire call.

    raise ValueError(
        f"Simulator failed to return valid JSON after {1 + MAX_RETRIES} "
        f"attempts. Last error: {last_error}"
    )


async def call_synthesizer(
    full_cascade_context: str,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Synthesize a concrete product / service idea from the full cascade
    trace.

    Returns dict with keys: title, mechanism, target_user,
    core_value_prop, assumption_chain, mvt, adjacent_analogues
    """
    user = (
        f"Full cascade trace (assumption → inversion → simulation at "
        f"each depth):\n\n{full_cascade_context}\n\n"
        f"Synthesize a single concrete idea from this cascade."
    )

    return await safe_llm_call(
        SYNTHESIZER_SYSTEM, user, model=model, temperature=0.8
    )
