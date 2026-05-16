"""Curated demo idea packs for the Conceptial streaming endpoint.

These packs power the wow-moment demo flows. They're written for the kind of
creative-operations work that brand designers, agency creatives, and marketing
directors actually do (visual identity, campaign concepts, packaging directions,
brand naming) — not generic business strategy.

Each pack has 8 ideas, deliberately spread across stylistic axes so anti-mode
exclusion has somewhere to push the next regeneration toward.

The streaming layer round-robins through ``CONCEPTIAL_PACKS`` based on a hash
of (prompt + project_id). The same prompt on the same project keeps pulling
from the same pool; the IdeaMemoryStore filters out anything already shown so
repeated regenerations widen the explored space.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoIdea:
    summary: str
    content: str
    pack: str


# 1) Visual identity / logo directions ---------------------------------------

_PACK_VISUAL_IDENTITY: list[DemoIdea] = [
    DemoIdea(
        pack="visual-identity",
        summary="Cardinal Lattice",
        content=(
            "A modular grid built from intersecting line-weights — the Stanford "
            "cardinal red rendered as eight overlapping crimson strokes that "
            "form a tree silhouette when viewed at distance and a circuit diagram "
            "when read close-up. Pairs with a single sans-serif wordmark in matte "
            "black, set at -10 tracking. Works in monochrome, embroidery, and as "
            "a single-color emboss on letterpress invites."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Hoover Bracket",
        content=(
            "A precise architectural bracket motif borrowed from the Hoover "
            "Tower's bell housing — abstracted into a pair of opposing right "
            "angles framing the wordmark. Renders with a 1.5pt stroke in "
            "navy or cardinal. Strong embossed at small sizes. Conveys "
            "institutional credibility without leaning on a generic shield."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Quad Topographic",
        content=(
            "A topographic-line rendering of the Main Quad's iconic arches, "
            "stripped to 5 isolines and rotated 30°. The contour lines double "
            "as soundwaves when animated. Sand and ink-black on cream paper "
            "for letterpress; cardinal red and white for digital. Pairs with "
            "Caslon for editorial weight."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Eleven Disciplines Mosaic",
        content=(
            "Eleven small icons — one per engineering discipline at Stanford — "
            "tessellated into a hexagonal mark. Each icon is reducible to a "
            "single weight: a pendulum for mechanical, a node for EE, a helix "
            "for bioE, etc. Together they form a single recognisable badge; "
            "individually they're a sticker set for sub-club merch."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Sequoia Section",
        content=(
            "A cross-section through a sequoia trunk rendered as concentric "
            "circles in graphite — each ring labelled with a Stanford founding "
            "year, recent breakthrough, or alumni achievement. Reads as a "
            "scientific diagram and a legacy statement at the same time. "
            "Best as a foil-stamped frontispiece or a poster."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Palo Sketch",
        content=(
            "A pen-and-ink sketch of the El Palo Alto redwood, drawn in a "
            "single unbroken line, then turned into the brand's primary "
            "logomark. The unbroken-line constraint forces a deliberately "
            "imperfect, hand-drawn feel. Pairs with a typewriter-style monospace "
            "wordmark. Conveys craft, not corporate."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Maker Workbench",
        content=(
            "A flat-illustrated overhead workbench — soldering iron, "
            "circuit board, half-built widget — used as a recurring "
            "key art across collateral. The logomark is one of the tools "
            "on the bench, rendered isometrically. Conveys hands-on "
            "engineering culture, distinct from the institutional polish."
        ),
    ),
    DemoIdea(
        pack="visual-identity",
        summary="Stanford Plate Negative",
        content=(
            "Renders the wordmark as a black-and-white photographic plate "
            "negative — a Cardinal-red wordmark inverted against a black "
            "background, with subtle film grain. Builds a documentary, "
            "research-archive feel. Excellent on dark merch and Instagram "
            "story templates."
        ),
    ),
]


# 2) Campaign concepts -------------------------------------------------------

_PACK_CAMPAIGN: list[DemoIdea] = [
    DemoIdea(
        pack="campaign",
        summary="Field Notes from Lab 47",
        content=(
            "A handwritten-letter campaign. Each piece in the series is a "
            "single hand-scrawled lab notebook page — coffee stains, marginalia, "
            "a half-finished equation — printed at A3 and pinned to walls in "
            "high-foot-traffic engineering corridors. The QR code in the margin "
            "leads to the club's open project list. Feels personal, scarce, "
            "intentionally not designed."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="The Apprentice Series",
        content=(
            "A six-week documentary podcast following first-year engineering "
            "students paired with senior makers. Each 20-minute episode focuses "
            "on one specific build — a sensor, a robot arm, a fabric circuit. "
            "Distribution through campus radio and the club's Substack. Cost: "
            "one good mic and 2 hours/week of editing."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="Build Night, Open House",
        content=(
            "A weekly Wednesday-evening open-house format. The club's main "
            "workshop is unlocked from 7–10pm, equipment is staffed, projects "
            "are visible. No agenda, no demos, just open access. Marketing "
            "is a single Instagram story each Tuesday and a chalk arrow on "
            "the sidewalk outside the building. Counts on the visceral pull "
            "of seeing other people make things."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="Defect Gallery",
        content=(
            "An anti-portfolio exhibit. Every project's first failed prototype "
            "is photographed in studio lighting, labelled with what went wrong "
            "and what got fixed, then mounted alongside the working version. "
            "Communicates that the path to mastery is mostly broken parts. "
            "Strong recruiting tool — students see they don't need to arrive "
            "perfect."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="Office Hours at the Coupa",
        content=(
            "A weekly informal-coffee format at Coupa Cafe. Two senior club "
            "members hold court for 90 minutes; anyone can sit down and ask "
            "anything technical. Slack thread captures the questions and seeds "
            "future workshop topics. Zero production cost, builds a reputation "
            "for accessibility. The presence is the marketing."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="One Brief Per Week",
        content=(
            "Every Monday the club publishes one tiny engineering brief — "
            "'design a 3D-printable replacement for X', 'redesign the campus "
            "garbage bin', 'build a single-button physical interface for Y'. "
            "Members submit by Friday; submissions are voted on Sunday. The "
            "best brief becomes a sticker. Builds rhythm and a public output "
            "stream."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="The Year One Map",
        content=(
            "A printed campus map handed out at orientation showing every "
            "lab, every maker space, every shop, and every faculty office that "
            "welcomes drop-ins. Folded into a poster on the reverse. Solves "
            "the most common new-student problem (not knowing where things "
            "are) and the club becomes the publisher of the answer."
        ),
    ),
    DemoIdea(
        pack="campaign",
        summary="Failure Friday Email",
        content=(
            "A weekly Friday newsletter, single-screen, single-story format. "
            "Each issue tells the story of one engineering failure in detail — "
            "what was attempted, what broke, what was learned. Industry "
            "post-mortems, historical case studies, member projects, all "
            "treated equally. Builds intellectual seriousness and a "
            "differentiated voice."
        ),
    ),
]


# 3) Packaging / product design directions -----------------------------------

_PACK_PACKAGING: list[DemoIdea] = [
    DemoIdea(
        pack="packaging",
        summary="Field-Issue Crate",
        content=(
            "A flat-pack corrugated crate stencil-printed with the club logo "
            "and a manifest of contents on the outside. Internal die-cuts "
            "hold a starter kit: breadboard, jumper wires, soldering iron, "
            "a one-page get-started guide. Mailable. Used as a recruit-out "
            "package for accepted incoming freshmen during the summer."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Spec Sheet First",
        content=(
            "Every project shipped with a one-page technical spec sheet on "
            "the front of the box — the actual schematic, dimensions, BOM "
            "and a QR to the build log. The product is treated as documented, "
            "open hardware. The spec sheet IS the marketing. No flowery copy."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Hand-Numbered Editions",
        content=(
            "Limited-run merch and project kits are hand-numbered with a "
            "stamp on the inside of the box, recording maker initials and "
            "the build date. Builds a craft-object feel even on mass-produced "
            "items. Excellent for sticker packs, board kits, and bottle "
            "openers as fundraising merch."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Pull-Tab Lab Notebook",
        content=(
            "A bound lab notebook handed to every new member, with "
            "perforated pull-tab pages along the binding. Each pull-tab "
            "is a one-line credit voucher — 'good for 1 hour of mentor "
            "time', 'good for 1 board fabrication', 'good for 1 free "
            "coffee from the club fridge'. Drives the use of resources "
            "without bureaucracy."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Anodised Aluminium Pin",
        content=(
            "A single-color anodised aluminium pin (cardinal red, machined, "
            "0.6mm thick) as the only physical brand object members ever "
            "carry. Conveys precision manufacturing. Functions as access "
            "credential for member-only spaces. The scarcity becomes the "
            "identity statement."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Open-Source Patch Set",
        content=(
            "Six embroidered patches — one for each of the club's main labs "
            "or sub-disciplines. Members earn them by shipping a project "
            "in that domain. Patches are physical proof of competence; "
            "the merch becomes a portfolio. Patches are downloadable as "
            "open-source SVG files so members can stitch their own variants."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Bento-Box Toolkit",
        content=(
            "A clear-acrylic bento-style box, partitioned, sold at cost to "
            "members for $40. Pre-loaded with the most-used components: "
            "Arduino, jumpers, sensors, a Mini breadboard, basic "
            "multimeter. Replaces the 'I don't have parts' excuse with "
            "a one-time barrier. Refill pouches sold at the door of "
            "Build Night."
        ),
    ),
    DemoIdea(
        pack="packaging",
        summary="Newsprint Project Reports",
        content=(
            "End-of-quarter project reports printed on newsprint, broadsheet "
            "format, in classic editorial layout. Distributed for free at "
            "central campus locations. The newsprint medium signals "
            "documentary seriousness without the cost of glossy stock. "
            "Easily archived; immediately recognisable."
        ),
    ),
]


CONCEPTIAL_PACKS: dict[str, list[DemoIdea]] = {
    "visual-identity": _PACK_VISUAL_IDENTITY,
    "campaign": _PACK_CAMPAIGN,
    "packaging": _PACK_PACKAGING,
}


def all_demo_ideas() -> list[DemoIdea]:
    """Flatten every pack into a single list."""
    out: list[DemoIdea] = []
    for pack in CONCEPTIAL_PACKS.values():
        out.extend(pack)
    return out


__all__ = [
    "CONCEPTIAL_PACKS",
    "DemoIdea",
    "all_demo_ideas",
]
