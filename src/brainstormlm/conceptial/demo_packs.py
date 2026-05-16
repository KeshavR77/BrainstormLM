"""Curated demo idea packs for the Conceptial streaming endpoint.

These packs power the wow-moment demo flows for creative-operations audiences.
Each idea is written the way a brand designer, art director, or creative
strategist would write it — concrete materials, sensory detail, references
to typography, photography, palette, and production craft.

The packs span three axes of creative work:

- ``brand-identity``  — logomark, wordmark, type system, brand world
- ``campaign-craft`` — campaign concept, editorial idea, launch motif
- ``object-design``  — packaging, product, merch, physical object craft

The streaming source rotates through all packs deterministically by
hash(prompt + project_id). Per-project IdeaMemoryStore exclusion keeps
regenerations from repeating, so the audience sees the project's memory
visibly grow on the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoIdea:
    summary: str
    content: str
    pack: str


# ── Brand identity / visual systems ──────────────────────────────────────────

_PACK_BRAND: list[DemoIdea] = [
    DemoIdea(
        pack="brand-identity",
        summary="Linen Folio",
        content=(
            "An identity built around a heavyweight linen-paper texture as the "
            "primary brand surface. The wordmark is set in a wide-letterspaced "
            "modern serif, foil-stamped in soft gold. Photography is shot on "
            "medium-format film, daylight only, with deep shadow and warm "
            "highlight. Colour palette: bone white, ink black, a single warm "
            "amber accent reserved for the call-to-action. Pairs with a "
            "monospaced support type for legal copy. Reads as quietly "
            "expensive, no exclamation marks."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Risograph Notebook",
        content=(
            "Built around the riso aesthetic — a 4-colour limited palette "
            "(fluorescent pink, navy, raw umber, off-white), visible halftone "
            "patterns, deliberate misregistration. Wordmark is a hand-drawn "
            "monoline. Photography is collaged from open-source archive plates. "
            "Reads as DIY, art-school, intentional. Best suited for an indie "
            "label that wants to look like a community press, not a startup."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Bauhaus Diagram",
        content=(
            "A logomark constructed from primary geometric forms — circle, "
            "square, triangle — overlaid in CMYK process colours with visible "
            "spot registration. The wordmark sits below in a precise grotesk. "
            "Editorial system uses 12-column grid, oversized numerals as "
            "section breaks, and one full-bleed photographic plate per spread. "
            "Conveys engineering rigor wrapped in artistic precision."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Soft Botanical",
        content=(
            "A hand-painted botanical motif at the centre of the mark — sage, "
            "yarrow, and chamomile — rendered in watercolour and digitised. "
            "Palette: warm cream, sage green, terracotta, a single deep "
            "umber as text colour. Wordmark is a custom-cut display serif "
            "with humanist proportions. Photography is overhead flat-lay, "
            "natural shadow, no studio gradients. Reads as apothecary, "
            "wellness, slow craft."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Acid Modernist",
        content=(
            "An identity built around bright neon gradients (acid green to "
            "electric magenta) layered over photographic black-and-white. "
            "Wordmark is set in a custom variable typeface that morphs weight "
            "across the brand surface. Editorial uses oversized pull-quotes, "
            "ragged line breaks, and asymmetric layouts. Reads as digital-"
            "native, late-90s aware, post-internet. Best for a brand that "
            "wants to feel intentionally now."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Stone Carved",
        content=(
            "Wordmark engraved into a single rectangular block — visible "
            "chisel marks, deep relief, photographed in raking side-light. "
            "Brand surfaces use lithographic-stone textures as background. "
            "Palette: wet slate, raw limestone, a single deep ochre. "
            "Typography pairs a deep-cut Trajan with a humanist sans. "
            "Reads as institutional, durable, civic. For brands that need "
            "to communicate permanence."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Postal Service",
        content=(
            "A system designed to look like a 1960s postal authority. Single "
            "geometric logomark — a stylised envelope reduced to two angles. "
            "Wordmark in a wide DIN-style sans, kerned tight. Brand surfaces "
            "use stamp-perforation edges and rubber-stamp date marks. "
            "Palette: deep navy, post-office red, a single warm grey for "
            "supporting copy. Reads as utilitarian, civic, dependable. "
            "Excellent for logistics or trust-led services."
        ),
    ),
    DemoIdea(
        pack="brand-identity",
        summary="Studio Pottery",
        content=(
            "Identity built around hand-thrown ceramic objects as the primary "
            "photography subject — each photographed singly, frontally, with "
            "the maker's thumbprint visible in the clay. Wordmark is hand-cut, "
            "letterpress-printed onto kraft labels glued onto product. Palette "
            "is dictated by glaze chemistry: oxblood, celadon, oatmeal, soot. "
            "Conveys handmade, slow, durable. Costly to scale; perfect for a "
            "small luxury label."
        ),
    ),
]


# ── Campaign craft ──────────────────────────────────────────────────────────

_PACK_CAMPAIGN: list[DemoIdea] = [
    DemoIdea(
        pack="campaign-craft",
        summary="The Open Studio Tour",
        content=(
            "A six-week campaign in which the brand opens its physical studio "
            "on a single afternoon each week. Each tour is led by a different "
            "team member (head of design, founder, lead developer). No agenda "
            "beyond 'walk through what we make and why.' Promotion is a "
            "simple weekly Instagram story and a chalkboard on the sidewalk. "
            "Costs effectively nothing. Generates word-of-mouth that paid "
            "media can't manufacture."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="One Object, Seven Stories",
        content=(
            "A campaign anchored on a single hero product, photographed seven "
            "different ways across seven different lifestyles and locations. "
            "Each shot becomes the visual for a single editorial story (a "
            "newsletter, a billboard, a press feature). Tells the audience: "
            "this object earns its place in many lives. Compresses the budget "
            "into one rigorous photo shoot instead of dilute multi-product "
            "rollouts."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="The Drop Letter",
        content=(
            "Each new product launch is announced via a single physical letter "
            "mailed to the most loyal 500 customers — handwritten address, "
            "wax seal, no digital duplication. The letter contains an access "
            "code valid for 48 hours. The exclusivity is small and real. "
            "Drives early sales, social sharing, and a fierce loyalty effect. "
            "Production cost: under $1,500 per launch."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="The Failure Edition",
        content=(
            "An annual print zine documenting every project the brand "
            "shipped — and every project that was killed before it shipped. "
            "Each killed project gets one spread: what was attempted, what "
            "went wrong, what was learned. Distributed free at retail and "
            "events. Counter-cultural; communicates intellectual seriousness; "
            "becomes a recruiting tool for the people you actually want."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="Pop-Up Library",
        content=(
            "A six-week installation in a major city — a free public reading "
            "room curated around the brand's intellectual lineage (books, "
            "objects, films, music). No products for sale. No staff in brand "
            "uniform. Open to the public during gallery hours. Becomes a "
            "destination, gets press, builds the kind of brand authority that "
            "performance marketing cannot purchase."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="Co-Author Series",
        content=(
            "Pair a member of the in-house creative team with an outside "
            "guest each quarter — a chef, a novelist, a musician, an architect — "
            "to produce a single co-authored object or moment. The collaboration "
            "is documented across all owned channels. Costs are modest; "
            "borrows credibility from adjacent disciplines; creates a body "
            "of work that compounds over years."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="The Quiet Newsletter",
        content=(
            "A monthly email, no images, no buttons, no calls to action. One "
            "essay per issue, written by a single person on the team, about "
            "something they're learning. Distributed only by direct invitation. "
            "The list grows by word-of-mouth. Conversion is high precisely "
            "because no one is asked to convert. The patience of the format "
            "is the message."
        ),
    ),
    DemoIdea(
        pack="campaign-craft",
        summary="The Field Recording",
        content=(
            "A single field-recording-style ambient audio loop — the sound of "
            "the brand's actual workspace — published as a continuous YouTube "
            "stream and Spotify session. No music, no voiceover, no marketing. "
            "Becomes a focus track for designers and writers who want a brand "
            "to live alongside their work. Builds an oddly intimate relationship "
            "with the audience without selling them anything."
        ),
    ),
]


# ── Object design / packaging ───────────────────────────────────────────────

_PACK_OBJECT: list[DemoIdea] = [
    DemoIdea(
        pack="object-design",
        summary="Field-Issue Kit",
        content=(
            "A flat-pack corrugated cardboard kit, stencil-printed in two "
            "spot colours. Inside: every part needed for first use, organised "
            "by die-cut compartments, labelled with serial numbers and a one-"
            "page reference card. Mailable as-is. Designed to be reassembled "
            "into a desk organiser after unboxing — the packaging becomes "
            "the second use."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Spec Sheet First",
        content=(
            "Every product ships with a one-page technical spec sheet adhered "
            "to the outside of the box. Real schematic, real dimensions, real "
            "bill of materials. The spec sheet IS the marketing. No flowery "
            "copy. Communicates engineering seriousness; gives the customer "
            "a reason to keep the packaging long enough to read it."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Glass Decanter Series",
        content=(
            "Primary product packaged in a single mouth-blown decanter, each "
            "with a hand-tied jute label noting the batch and bottling date. "
            "Decanters are returnable for a 20% future-order credit. The "
            "packaging becomes the brand object in the customer's home long "
            "after the product is consumed. Cost-justified by the repeat-"
            "purchase loop, not the unit margin."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Newsprint Wrap",
        content=(
            "Product ships wrapped in newsprint printed with the brand's "
            "quarterly editorial — recipes, essays, photographs by guest "
            "contributors. The wrap is the read; the read becomes a moment "
            "of unboxing pause. Newsprint is cheap, recyclable, signals "
            "documentary seriousness. Easily archived; quietly distinct from "
            "the glossy-stock norm."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Hand-Numbered Edition",
        content=(
            "Every package is hand-stamped with a sequential number and the "
            "initials of the person who packed it. Builds a craft-object feel "
            "into an otherwise mass-produced product. Excellent on small "
            "runs and as a fundraising or limited-edition tactic. The hand "
            "is the feature."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Refill Manifesto",
        content=(
            "Primary container is heavyweight glass or anodised aluminium, "
            "sold once. Refills ship in light kraft pouches at half the unit "
            "price. The brand becomes the lifetime container; every refill "
            "is a soft recommitment. Communicates sustainability without "
            "any extra messaging — the architecture of the product is the "
            "statement."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Cloth Sleeve",
        content=(
            "Product is wrapped in a single piece of woven cotton sleeve, "
            "sewn with the brand's wordmark on the inside fold. Sleeve "
            "doubles as a dust cover and a souvenir. Tactile, slow, "
            "intentionally undisposable. Costs ~$2/unit at scale; reads "
            "as a $20 add-on in customer perception."
        ),
    ),
    DemoIdea(
        pack="object-design",
        summary="Blueprint Tag",
        content=(
            "Each product carries a single hang-tag printed on tracing paper "
            "with the product's exploded technical drawing on the front and "
            "the maker's name on the back. The tag is detachable, archive-"
            "worthy. Customers who care about provenance keep them; "
            "customers who don't get a beautiful object regardless."
        ),
    ),
]


CONCEPTIAL_PACKS: dict[str, list[DemoIdea]] = {
    "brand-identity": _PACK_BRAND,
    "campaign-craft": _PACK_CAMPAIGN,
    "object-design": _PACK_OBJECT,
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
