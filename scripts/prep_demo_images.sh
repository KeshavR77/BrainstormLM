#!/usr/bin/env bash
# Prep the Demo page's image collage.
#
# - Reads source PNGs from $SRC (default ~/Downloads).
# - Center-crops the curated set to 1024x1024 squares.
# - Saves as compressed WebP at quality 75 (sips built-in on macOS).
# - Generates 8 SVG placeholders for the Conceptial 14B column.
#
# Output: frontend/public/demo/{gpt,anthropic,conceptial}/NN.webp|svg
#
# Re-runnable; overwrites prior outputs. Source PNGs in Downloads untouched.

set -euo pipefail

SRC="${SRC:-$HOME/Downloads}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/frontend/public/demo"

# Anthropic curation — drop anthropic3 (sleeve mockup) + anthropic8 (fabric patch)
ANTHROPIC_KEEP=(1 2 4 5 6 7 9 10)
GPT_KEEP=(1 2 3 4 5 6 7 8)

# Conceptial placeholder concept names — first 8 from
# artifacts/logo_faceoff/logo_faceoff.json
CONCEPT_NAMES=(
  "Engineer's Nexus"
  "3D Blueprint Vision"
  "Craftsmanship Convergence"
  "Quantum Gear Revolution"
  "Innovate Connect"
  "Inspire Mechanism"
  "Bridge to Innovation"
  "Inventive Horizon"
)

mkdir -p "$OUT/gpt" "$OUT/anthropic" "$OUT/conceptial"

# ── Helper: center-crop to square 1024x1024, encode as WebP @ q=75 ────────
# sips is on every macOS install; no extra deps required.
process_to_square_webp() {
  local input_png="$1"
  local output_webp="$2"
  local tmp
  tmp="$(mktemp -d)"
  # 1. Center-crop to square (uses the smaller dim, drops sides/top-bottom)
  cp "$input_png" "$tmp/in.png"
  local w h edge
  w=$(sips -g pixelWidth "$tmp/in.png" | awk '/pixelWidth/{print $2}')
  h=$(sips -g pixelHeight "$tmp/in.png" | awk '/pixelHeight/{print $2}')
  edge=$(( w < h ? w : h ))
  sips -c "$edge" "$edge" "$tmp/in.png" --out "$tmp/sq.png" > /dev/null
  # 2. Resize to 1024x1024
  sips -Z 1024 "$tmp/sq.png" --out "$tmp/r.png" > /dev/null
  # 3. Encode as WebP via cwebp (sips on this macOS doesn't support webp output)
  cwebp -quiet -q 75 -resize 1024 1024 "$tmp/r.png" -o "$output_webp"
  rm -rf "$tmp"
}

echo "→ Encoding GPT-image set (8 files)"
i=0
for n in "${GPT_KEEP[@]}"; do
  i=$((i + 1))
  src_path="$SRC/gpt${n}.png"
  out_path="$OUT/gpt/$(printf '%02d' "$i").webp"
  if [[ ! -f "$src_path" ]]; then
    echo "  ! missing $src_path — skipping"
    continue
  fi
  process_to_square_webp "$src_path" "$out_path"
  size_kb=$(( $(stat -f%z "$out_path") / 1024 ))
  echo "  ✓ gpt $(printf '%02d' "$i").webp  (${size_kb} KB)"
done

echo "→ Encoding Anthropic set (8 files; dropped anthropic3 + anthropic8)"
i=0
for n in "${ANTHROPIC_KEEP[@]}"; do
  i=$((i + 1))
  # anthropic10 lives at antrhopic10.png (typo in source)
  if [[ "$n" == "10" ]]; then
    src_path="$SRC/antrhopic10.png"
  else
    src_path="$SRC/anthropic${n}.png"
  fi
  out_path="$OUT/anthropic/$(printf '%02d' "$i").webp"
  if [[ ! -f "$src_path" ]]; then
    echo "  ! missing $src_path — skipping"
    continue
  fi
  process_to_square_webp "$src_path" "$out_path"
  size_kb=$(( $(stat -f%z "$out_path") / 1024 ))
  echo "  ✓ anthropic $(printf '%02d' "$i").webp  (${size_kb} KB)"
done

echo "→ Generating Conceptial SVG placeholders (8 files)"
for i in 1 2 3 4 5 6 7 8; do
  idx=$((i - 1))
  name="${CONCEPT_NAMES[$idx]}"
  out_path="$OUT/conceptial/$(printf '%02d' "$i").svg"
  # Escape XML special chars in concept name
  escaped="${name//&/&amp;}"
  escaped="${escaped//</&lt;}"
  escaped="${escaped//>/&gt;}"
  cat > "$out_path" <<EOF
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" preserveAspectRatio="xMidYMid meet">
  <defs>
    <linearGradient id="g$i" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#F4EFE6"/>
      <stop offset="1" stop-color="#EDE6D9"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" fill="url(#g$i)"/>
  <rect x="32" y="32" width="960" height="960" rx="32" ry="32"
        fill="none" stroke="#C9BFAE" stroke-width="2" stroke-dasharray="8 10"/>
  <text x="512" y="160" text-anchor="middle"
        font-family="'JetBrains Mono', ui-monospace, monospace"
        font-size="28" letter-spacing="6" fill="#8A7E69">
    CONCEPTIAL · 14B
  </text>
  <text x="512" y="520" text-anchor="middle"
        font-family="'Fraunces', Georgia, serif"
        font-style="italic"
        font-size="56" fill="#1F1B14"
        font-weight="420">
    ${escaped}
  </text>
  <text x="512" y="600" text-anchor="middle"
        font-family="'Inter', system-ui, sans-serif"
        font-size="22" fill="#7A6F5E">
    Image rendering pending
  </text>
  <g transform="translate(512 820)" fill="none" stroke="#C9BFAE" stroke-width="2">
    <circle cx="-32" cy="0" r="6" fill="#C9BFAE"/>
    <circle cx="0" cy="0" r="6" fill="#C9BFAE"/>
    <circle cx="32" cy="0" r="6" fill="#C9BFAE"/>
  </g>
</svg>
EOF
  echo "  ✓ conceptial $(printf '%02d' "$i").svg  (${name})"
done

echo
echo "Done. Total size:"
du -sh "$OUT" 2>/dev/null
ls -la "$OUT"/{gpt,anthropic,conceptial} 2>/dev/null | grep -v "^total\|^d" | head -30
