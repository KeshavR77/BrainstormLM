/**
 * ScrollTree.tsx — Scroll-driven idea map.
 *
 * "Studio Midnight" palette: muted gold (#c9a55c), teal (#5eead4) selection.
 * Refined strokes, subtle glows, premium pill styling.
 */

import { useMemo } from 'react';

interface ScrollTreeProps { progress: number; className?: string }

function band(t: number, lo: number, hi: number) {
  return Math.max(0, Math.min(1, (t - lo) / (hi - lo)));
}
function ease(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

interface N { id: string; x: number; y: number; label: string; type: 'seed' | 'idea' | 'chosen' | 'deep' }
interface E { from: string; to: string }

const CX = 500, W = 1000, H = 480;

const NODES: N[] = [
  { id: 's',  x: CX,       y: 55,  label: 'How do we grow to 10M users?',    type: 'seed' },
  { id: 'a',  x: CX - 280, y: 200, label: 'Bottoms-up viral loop',   type: 'idea' },
  { id: 'b',  x: CX,       y: 200, label: 'Platform before product', type: 'chosen' },
  { id: 'c',  x: CX + 280, y: 200, label: 'Geographic arbitrage',    type: 'idea' },
  { id: 'd1', x: CX - 200, y: 360, label: 'API marketplace',         type: 'deep' },
  { id: 'd2', x: CX,       y: 360, label: 'White-label offering',    type: 'deep' },
  { id: 'd3', x: CX + 200, y: 360, label: 'Partner ecosystem',       type: 'deep' },
];

const EDGES: E[] = [
  { from: 's', to: 'a' }, { from: 's', to: 'b' }, { from: 's', to: 'c' },
  { from: 'b', to: 'd1' }, { from: 'b', to: 'd2' }, { from: 'b', to: 'd3' },
];

const NM = Object.fromEntries(NODES.map(n => [n.id, n]));

/* ── Colors ──────────────────────────────────────────────────────────── */
const GOLD     = { r: 201, g: 165, b: 92  };   // #c9a55c
const TEAL     = { r: 94,  g: 234, b: 212 };   // #5eead4
const SURFACE  = 'rgba(16,16,24,0.9)';
const SURFACE_SEL = 'rgba(14,22,20,0.92)';

function rgba(c: typeof GOLD, a: number) { return `rgba(${c.r},${c.g},${c.b},${a})`; }

export function ScrollTree({ progress, className }: ScrollTreeProps) {
  const a = useMemo(() => ({
    seed:  ease(band(progress, 0.02, 0.10)),
    e1:    ease(band(progress, 0.08, 0.26)),
    nA:    ease(band(progress, 0.18, 0.28)),
    nB:    ease(band(progress, 0.22, 0.32)),
    nC:    ease(band(progress, 0.26, 0.36)),
    sel:   ease(band(progress, 0.40, 0.52)),
    e2:    ease(band(progress, 0.50, 0.66)),
    nD1:   ease(band(progress, 0.58, 0.66)),
    nD2:   ease(band(progress, 0.62, 0.70)),
    nD3:   ease(band(progress, 0.66, 0.74)),
    bloom: ease(band(progress, 0.72, 0.86)),
    // Flow: a travelling bead along each edge as it draws
    flow1: band(progress, 0.08, 0.30),
    flow2: band(progress, 0.50, 0.70),
  }), [progress]);

  const opMap: Record<string, number> = {
    s: a.seed, a: a.nA, b: a.nB, c: a.nC, d1: a.nD1, d2: a.nD2, d3: a.nD3,
  };

  function curve(f: N, t: N) {
    const y1 = f.y + 22, y2 = t.y - 22, m = (y1 + y2) / 2;
    return `M${f.x},${y1} C${f.x},${m} ${t.x},${m} ${t.x},${y2}`;
  }
  function len(f: N, t: N) { return Math.hypot(t.x - f.x, t.y - f.y) * 1.3; }

  function dims(n: N): { pw: number; ph: number } {
    if (n.type === 'seed') return { pw: 320, ph: 54 };
    if (n.type === 'chosen') return { pw: 200, ph: 46 };
    if (n.type === 'deep') return { pw: 180, ph: 42 };
    return { pw: 185, ph: 46 };
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} preserveAspectRatio="xMidYMid meet">
      <defs>
        <radialGradient id="bg-g">
          <stop offset="0%" stopColor={rgba(GOLD, 0.03)} />
          <stop offset="100%" stopColor={rgba(GOLD, 0)} />
        </radialGradient>
      </defs>
      <ellipse cx={CX} cy={H * 0.45} rx={340} ry={190} fill="url(#bg-g)" opacity={a.seed} />

      {/* Edges */}
      {EDGES.map(e => {
        const f = NM[e.from], t = NM[e.to], d = curve(f, t), l = len(f, t);
        const isL1 = e.from === 's';
        const draw = isL1 ? a.e1 : a.e2;
        if (draw < 0.01) return null;
        const isSel = e.from === 'b' && a.sel > 0.3;
        const col = isSel ? TEAL : GOLD;

        // Travelling bead along the path — only while the edge is "alive"
        const flow = isL1 ? a.flow1 : a.flow2;
        const showBead = flow > 0.01 && flow < 0.98 && draw > 0.15;
        const beadOffset = l * (1 - (flow % 1));

        return (
          <g key={`${e.from}-${e.to}`}>
            {isSel && (
              <path d={d} fill="none" stroke={rgba(col, 0.08)} strokeWidth={8} strokeLinecap="round"
                strokeDasharray={l} strokeDashoffset={l * (1 - draw)} style={{ filter: 'blur(6px)' }} />
            )}
            <path d={d} fill="none" stroke={rgba(col, isSel ? 0.35 : 0.18)} strokeWidth={1.5}
              strokeLinecap="round" strokeDasharray={l} strokeDashoffset={l * (1 - draw)} />
            {showBead && (
              <path d={d} fill="none" stroke={rgba(col, 0.9)} strokeWidth={2.2} strokeLinecap="round"
                strokeDasharray={`3 ${l}`} strokeDashoffset={beadOffset}
                style={{ filter: 'drop-shadow(0 0 4px ' + rgba(col, 0.6) + ')' }} />
            )}
          </g>
        );
      })}

      {/* Nodes */}
      {NODES.map(n => {
        const op = opMap[n.id] ?? 0;
        if (op < 0.01) return null;
        const isSeed = n.type === 'seed', isSel = n.type === 'chosen', sg = isSel ? a.sel : 0;
        const { pw, ph } = dims(n);
        const rx = ph / 2;

        const fill = isSel ? SURFACE_SEL : SURFACE;
        const col = isSel ? TEAL : GOLD;
        const strokeA = isSeed ? 0.2 + op * 0.2 : isSel ? 0.1 + sg * 0.35 : 0.04 + op * 0.04;
        const stroke = isSeed ? rgba(GOLD, strokeA) : isSel ? rgba(TEAL, strokeA) : `rgba(255,255,255,${strokeA})`;

        const textCol = isSeed ? rgba(GOLD, 0.85)
          : isSel ? rgba(TEAL, 0.5 + sg * 0.4)
          : `rgba(237,235,232,${0.4 + op * 0.15})`;

        return (
          <g key={n.id} opacity={op} style={{ transform: `translateY(${(1 - op) * 5}px)` }}>
            {/* Glow */}
            {(isSeed || (isSel && sg > 0.2)) && (
              <ellipse cx={n.x} cy={n.y} rx={pw * 0.4} ry={ph * 0.8}
                fill={rgba(col, 0.05)} opacity={isSeed ? 0.6 : sg * 0.6}
                style={{ filter: 'blur(20px)' }} />
            )}
            {/* Pill */}
            <rect x={n.x - pw / 2} y={n.y - ph / 2} width={pw} height={ph} rx={rx}
              fill={fill} stroke={stroke} strokeWidth={isSel && sg > 0.5 ? 1.5 : 1} />
            {/* Label */}
            <text x={n.x} y={n.y + (isSeed ? 6 : 5)} textAnchor="middle" fill={textCol}
              fontSize={isSeed ? 16 : 14} fontWeight={isSeed ? 500 : 400}
              fontFamily="Inter, system-ui, sans-serif">{n.label}</text>
            {/* SELECTED badge */}
            {isSel && sg > 0.3 && (
              <g opacity={sg}>
                <rect x={n.x - 38} y={n.y + ph / 2 + 8} width={76} height={22} rx={11}
                  fill={rgba(TEAL, 0.06)} stroke={rgba(TEAL, 0.2)} strokeWidth={0.7} />
                <text x={n.x} y={n.y + ph / 2 + 23} textAnchor="middle" fill={rgba(TEAL, 0.85)}
                  fontSize={10} fontWeight={600} fontFamily="Inter, system-ui, sans-serif" letterSpacing="0.08em">SELECTED</text>
              </g>
            )}
            {/* Dots */}
            {!isSeed && <circle cx={n.x} cy={n.y - ph / 2} r={2}
              fill={isSel && sg > 0.5 ? rgba(TEAL, 0.5) : rgba(GOLD, 0.3)} />}
            <circle cx={n.x} cy={n.y + ph / 2} r={2}
              fill={isSel && sg > 0.5 ? rgba(TEAL, 0.5) : rgba(GOLD, 0.3)} />
          </g>
        );
      })}

      {/* Final bloom */}
      {a.bloom > 0 && (
        <ellipse cx={CX} cy={280} rx={220} ry={120} fill={rgba(TEAL, 0.03)}
          opacity={a.bloom} style={{ filter: 'blur(40px)' }} />
      )}
    </svg>
  );
}
