/**
 * HeroTreeGrowth — hero idea map.
 *
 * A composed "live workspace" that sells what Conceptial does in 3 seconds:
 *   prompt drops in → branches grow with visible edges → cursor lands on the
 *   chosen branch → it scales up + glows → leaves cascade beneath it.
 *
 * Everything is internal: window chrome, grid backdrop, cursor, edges, cards,
 * footer telemetry. The wrapper only provides the outer rounded rectangle.
 */
import { motion, useReducedMotion } from 'framer-motion';
import { useEffect, useMemo, useState } from 'react';
import { easeOut } from '../../styles/motion';

interface Node {
  id:         string;
  parent:     string | null;
  depth:      0 | 1 | 2;
  archetype?: string;
  hue:        number | 'brand';
  title:      string;
  body?:      string;
  score?:     string;
  selected?:  boolean;
  x:          number;
  y:          number;
}

const ROOT_TEXT = 'How might onboarding reach the aha moment in 60s?';

const NODES: Node[] = [
  { id: 'root', parent: null,   depth: 0, hue: 'brand',
    title: ROOT_TEXT, x: 50, y: 16 },

  { id: 'a',    parent: 'root', depth: 1, hue: 280,
    archetype: 'User-led',   score: 'V 1.41',
    title: 'Instant personal payoff',
    body:  'Start from the user\'s goal.',
    x: 16, y: 54 },

  { id: 'b',    parent: 'root', depth: 1, hue: 'brand',
    archetype: 'Trust-led',  score: 'V 1.46', selected: true,
    title: 'Progressive trust',
    body:  'Prove value before setup.',
    x: 50, y: 54 },

  { id: 'c',    parent: 'root', depth: 1, hue: 200,
    archetype: 'Social-led', score: 'V 1.38',
    title: 'Social proof first',
    body:  'Lead with peer outcomes.',
    x: 84, y: 54 },

  { id: 'b1',   parent: 'b',    depth: 2, hue: 'brand',
    archetype: 'Sandbox',
    title: 'Sample-data sandbox',
    score: 'V 1.51',
    x: 36, y: 86 },

  { id: 'b2',   parent: 'b',    depth: 2, hue: 'brand',
    archetype: 'Receipt',
    title: 'Outcome receipt',
    score: 'V 1.44',
    x: 64, y: 86 },
];

const byId = Object.fromEntries(NODES.map((n) => [n.id, n])) as Record<string, Node>;
const SELECTED_ID = 'b';

function hueCss(h: Node['hue'], opacity = 1): string {
  if (h === 'brand') return opacity === 1 ? 'var(--brand)' : `rgba(225, 29, 116, ${opacity})`;
  return `oklch(0.62 0.20 ${h} / ${opacity})`;
}
function hueSoftCss(h: Node['hue']): string {
  if (h === 'brand') return 'rgba(225, 29, 116, 0.08)';
  return `oklch(0.96 0.04 ${h})`;
}

function edgePath(p: Node, c: Node, anchorOffset = 6): string {
  const dy = c.y - p.y;
  return `M ${p.x},${p.y + anchorOffset} C ${p.x},${p.y + dy * 0.55} ${c.x},${c.y - dy * 0.35} ${c.x},${c.y - anchorOffset}`;
}

export function HeroTreeGrowth({ className = '' }: { className?: string }) {
  const reduced = useReducedMotion();
  const [typed, setTyped] = useState(reduced ? ROOT_TEXT.length : 0);
  const rootReady = typed >= ROOT_TEXT.length;

  /* Typewriter */
  useEffect(() => {
    if (reduced) return;
    let i = 0;
    let t: number;
    const tick = () => {
      i += 1;
      setTyped(i);
      if (i < ROOT_TEXT.length) {
        const ch = ROOT_TEXT[i - 1];
        const wait = ch === ' ' ? 18 : ch === '?' ? 90 : 22 + Math.random() * 18;
        t = window.setTimeout(tick, wait);
      }
    };
    t = window.setTimeout(tick, 850);
    return () => window.clearTimeout(t);
  }, [reduced]);


  /* Timing constants — each stage is a clear, perceivable beat */
  const T = {
    rootCard:     0.5,
    branchEdges:  1.8,
    branchCards:  2.05,
    leafEdges:    3.0,
    leafCards:    3.2,
    idle:         3.6,
  };

  const branchIdx = useMemo(() => {
    const m: Record<string, number> = {};
    NODES.filter((n) => n.depth === 1).forEach((n, i) => { m[n.id] = i; });
    return m;
  }, []);
  const leafIdx = useMemo(() => {
    const m: Record<string, number> = {};
    NODES.filter((n) => n.depth === 2).forEach((n, i) => { m[n.id] = i; });
    return m;
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.985 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.7, ease: easeOut }}
      className={`relative w-full h-full ${className}`}
      aria-hidden="true"
    >
      {/* ── Faint grid backdrop + soft vignette focusing eye on center ── */}
      <div
        className="absolute inset-0 opacity-70 pointer-events-none"
        style={{
          backgroundImage:
            `linear-gradient(to right, rgba(22,19,24,0.045) 1px, transparent 1px),
             linear-gradient(to bottom, rgba(22,19,24,0.045) 1px, transparent 1px)`,
          backgroundSize: '22px 22px',
        }}
      />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at 50% 52%, transparent 0%, transparent 35%, rgba(244, 239, 230, 0.55) 85%)',
        }}
      />


      {/* ── Edges layer ───────────────────────────────────── */}
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full overflow-visible">
        <defs>
          <marker id="arrowSel" viewBox="0 0 8 8" refX="5" refY="4" markerWidth="4.5" markerHeight="4.5" orient="auto">
            <path d="M0,0.5 L7,4 L0,7.5 L1.5,4 z" fill="var(--brand)" />
          </marker>
          <linearGradient id="selGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%"  stopColor="var(--brand)" stopOpacity="0.5" />
            <stop offset="55%" stopColor="var(--brand)" stopOpacity="0.95" />
            <stop offset="100%" stopColor="var(--brand)" stopOpacity="1" />
          </linearGradient>
          <filter id="selGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="0.6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {NODES.filter((n) => n.parent).map((n) => {
          const parent = byId[n.parent!];
          const isSelectedPath = parent.id === SELECTED_ID || n.id === SELECTED_ID;
          const startT = n.depth === 1 ? T.branchEdges : T.leafEdges;
          const dur = isSelectedPath ? 0.85 : 0.7;
          const d = edgePath(parent, n);

          return (
            <g key={`e-${n.id}`}>
              {/* Soft halo under the selected path so it visually pops on cream */}
              {isSelectedPath && (
                <motion.path
                  d={d}
                  fill="none"
                  stroke="var(--brand)"
                  strokeWidth={3}
                  strokeLinecap="round"
                  opacity={0.12}
                  initial={{ pathLength: 0 }}
                  animate={rootReady ? { pathLength: 1 } : {}}
                  transition={{ duration: dur, ease: easeOut, delay: startT }}
                />
              )}
              <motion.path
                d={d}
                fill="none"
                stroke={isSelectedPath ? 'url(#selGrad)' : hueCss(n.hue)}
                strokeWidth={isSelectedPath ? 1.2 : 0.7}
                strokeLinecap="round"
                strokeDasharray={isSelectedPath ? undefined : '0.6 1.4'}
                markerEnd={isSelectedPath ? 'url(#arrowSel)' : undefined}
                filter={isSelectedPath ? 'url(#selGlow)' : undefined}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={rootReady ? { pathLength: 1, opacity: isSelectedPath ? 1 : 0.5 } : {}}
                transition={{
                  pathLength: { duration: dur, ease: easeOut, delay: startT },
                  opacity:    { duration: 0.3, delay: startT },
                }}
              />
              {/* Comet — small magenta dot that travels along the selected edge.
                  Repeats on a slow loop after the initial pass so the path stays alive. */}
              {isSelectedPath && (
                <motion.circle
                  r={0.7}
                  fill="var(--brand)"
                  initial={{ opacity: 0, offsetDistance: '0%' } as any}
                  animate={rootReady
                    ? ({ opacity: [0, 1, 1, 0, 0, 1, 1, 0],
                        offsetDistance: ['0%', '20%', '85%', '100%', '0%', '20%', '85%', '100%'] } as any)
                    : {}}
                  transition={{
                    duration: 6,
                    delay: startT,
                    ease: 'linear',
                    repeat: Infinity,
                    repeatType: 'loop',
                    times: [0, 0.06, 0.42, 0.48, 0.52, 0.58, 0.94, 1],
                  }}
                  style={{ offsetPath: `path('${d}')`, filter: 'drop-shadow(0 0 1.5px var(--brand))' } as React.CSSProperties}
                />
              )}
            </g>
          );
        })}
      </svg>


      {/* ── Nodes ─────────────────────────────────────────── */}
      {NODES.map((n) => {
        const isRoot   = n.depth === 0;
        const isBranch = n.depth === 1;
        const isLeaf   = n.depth === 2;
        const sibIdx   = isBranch ? branchIdx[n.id] : isLeaf ? leafIdx[n.id] : 0;
        const delay    = isRoot
          ? T.rootCard
          : isBranch
            ? T.branchCards + sibIdx * 0.08
            : T.leafCards + sibIdx * 0.08;

        const wcls = isRoot ? 'w-[50%]' : 'w-[24%]';
        const dimmed = isBranch && !n.selected;

        /* Outer div handles centering via inline transform (NOT a Tailwind
           class, so framer-motion's `animate.scale/y` can't override it).
           Inner motion.div handles the entrance + lift animation only. */
        return (
          <div
            key={n.id}
            className={`absolute ${wcls}`}
            style={{
              left: `${n.x}%`,
              top: `${n.y}%`,
              transform: 'translate(-50%, -50%)',
              zIndex: n.selected ? 5 : 1,
            }}
          >
            <motion.div
              initial={{ opacity: 0, y: 16, scale: 0.85 }}
              animate={{
                opacity: dimmed ? 0.55 : 1,
                y: n.selected ? -3 : 0,
                scale: n.selected ? 1.05 : isBranch ? 0.96 : 1,
              }}
              transition={{
                duration: 0.6,
                ease: [0.34, 1.45, 0.64, 1],
                delay,
              }}
            >
              {isRoot
                ? <RootCard typed={typed} reduced={!!reduced} />
                : isBranch
                  ? <BranchCard node={n} idleAt={T.idle} reduced={!!reduced} />
                  : <LeafCard   node={n} />}
            </motion.div>
          </div>
        );
      })}


      <style>{`
        @keyframes caret-blink { 50% { opacity: 0; } }
        @keyframes selRingPulse {
          0%, 100% { box-shadow: 0 0 0 2px var(--brand), 0 6px 22px rgba(225,29,116,0.22); }
          50%      { box-shadow: 0 0 0 2px var(--brand), 0 8px 32px rgba(225,29,116,0.42); }
        }
      `}</style>
    </motion.div>
  );
}


/* ─────────────────────────────────────────────────────────
   Card sub-components
   ───────────────────────────────────────────────────────── */

function RootCard({ typed, reduced }: { typed: number; reduced: boolean }) {
  return (
    <div
      className="relative rounded-xl bg-paper border border-[var(--ink-line)] px-4 py-3"
      style={{ boxShadow: '0 1px 0 rgba(22,19,24,0.04), 0 10px 28px rgba(22,19,24,0.07), 0 0 0 1px rgba(22,19,24,0.04)' }}
    >
      <div className="flex items-center gap-1.5 mb-2">
        <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-brand/10 border border-brand/20">
          <span className="h-1 w-1 rounded-full bg-brand" />
          <span className="text-[9px] uppercase tracking-[0.16em] text-brand font-medium">Prompt</span>
        </span>
        <span className="ml-auto text-[9.5px] text-ink-3 font-mono tabular-nums">root · d0</span>
      </div>
      <div
        className="font-display text-ink-0 text-[14.5px] leading-snug tracking-[-0.012em] min-h-[42px]"
        style={{ fontVariationSettings: '"wght" 500, "opsz" 40' }}
      >
        {ROOT_TEXT.slice(0, typed)}
        {!reduced && typed < ROOT_TEXT.length && (
          <span
            className="inline-block w-[1.4ch] align-baseline ml-0.5"
            style={{
              background: 'var(--brand)',
              height: '0.9em',
              opacity: 0.9,
              animation: 'caret-blink 1s steps(2) infinite',
            }}
          />
        )}
      </div>
      <div className="mt-2.5 pt-2 border-t border-[var(--ink-line-soft)] flex items-center justify-between">
        <span className="text-[10px] text-ink-2 font-medium">6 archetypes</span>
        <span className="text-[10px] font-mono text-ink-3 tabular-nums inline-flex items-center gap-1">
          <span className="h-1 w-1 rounded-full bg-brand" />
          3 explored
        </span>
      </div>
    </div>
  );
}

function BranchCard({ node, idleAt, reduced }: { node: Node; idleAt: number; reduced: boolean }) {
  const c = hueCss(node.hue);
  const soft = hueSoftCss(node.hue);
  const isSel = !!node.selected;

  return (
    <div
      className="relative rounded-lg border px-3 py-2.5"
      style={{
        background: isSel
          ? 'linear-gradient(180deg, rgba(255, 217, 232, 0.55) 0%, rgba(255, 255, 255, 0.95) 100%)'
          : 'var(--paper)',
        borderColor: isSel ? 'var(--brand)' : 'var(--ink-line)',
        boxShadow: isSel
          ? '0 0 0 2px var(--brand), 0 12px 32px rgba(225, 29, 116, 0.28), 0 1px 0 rgba(255,255,255,0.6) inset'
          : '0 1px 0 rgba(22,19,24,0.03), 0 3px 10px rgba(22,19,24,0.05)',
        animation: isSel && !reduced
          ? `selRingPulse 2.6s ease-in-out infinite ${idleAt}s`
          : undefined,
      }}
    >
      {/* Selected pill */}
      {isSel && (
        <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 inline-flex items-center gap-1 rounded-full bg-brand text-white text-[8.5px] font-bold uppercase tracking-[0.14em] px-2 py-0.5 shadow-[0_4px_14px_rgba(225,29,116,0.45)] whitespace-nowrap">
          <span className="h-1 w-1 rounded-full bg-white" />
          Your path
        </span>
      )}

      <div className="flex items-center gap-1.5 mb-1.5">
        <span className="h-1.5 w-1.5 rounded-full shrink-0 ring-2" style={{ background: c, '--tw-ring-color': isSel ? 'rgba(225,29,116,0.18)' : 'transparent' } as React.CSSProperties} />
        <span
          className="text-[9px] uppercase tracking-[0.14em] truncate font-semibold"
          style={{ color: isSel ? 'var(--brand-deep)' : 'var(--ink-2)' }}
        >
          {node.archetype}
        </span>
        {node.score && (
          <span
            className="ml-auto font-mono text-[9px] tabular-nums px-1 py-[1px] rounded"
            style={{ background: soft, color: isSel ? 'var(--brand-deep)' : 'var(--ink-1)' }}
          >
            {node.score}
          </span>
        )}
      </div>
      <div
        className={`text-[12.5px] font-medium leading-snug tracking-[-0.005em] mb-1 ${isSel ? 'text-ink-0' : 'text-ink-0'}`}
        style={{ fontVariationSettings: isSel ? '"wght" 600' : '"wght" 540' }}
      >
        {node.title}
      </div>
      {node.body && (
        <div className="text-[10.5px] text-ink-2 leading-snug">{node.body}</div>
      )}
      {isSel && (
        <div className="mt-2 pt-1.5 border-t border-brand/15 flex items-center justify-between">
          <span className="text-[9px] uppercase tracking-[0.14em] font-semibold text-brand inline-flex items-center gap-1">
            <span className="h-1 w-1 rounded-full bg-brand animate-pulse" style={{ animationDuration: '1.6s' }} />
            Exploring · 2 leaves
          </span>
          <span className="text-[10px] text-brand font-mono">↓</span>
        </div>
      )}
    </div>
  );
}

function LeafCard({ node }: { node: Node }) {
  const c = hueCss(node.hue);
  const soft = hueSoftCss(node.hue);
  return (
    <div
      className="relative rounded-md border px-2.5 py-2 backdrop-blur-sm"
      style={{
        background: 'linear-gradient(180deg, rgba(255, 217, 232, 0.25) 0%, rgba(255, 255, 255, 0.92) 100%)',
        borderColor: 'rgba(225, 29, 116, 0.22)',
        boxShadow: '0 1px 0 rgba(22,19,24,0.03), 0 4px 14px rgba(225, 29, 116, 0.10)',
      }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="h-1 w-1 rounded-full shrink-0" style={{ background: c }} />
        <span className="text-[8.5px] uppercase tracking-[0.14em] truncate font-semibold" style={{ color: 'var(--brand-deep)' }}>
          {node.archetype}
        </span>
        {node.score && (
          <span
            className="ml-auto font-mono text-[8.5px] tabular-nums px-1 rounded"
            style={{ background: soft, color: 'var(--brand-deep)' }}
          >
            {node.score}
          </span>
        )}
      </div>
      <div
        className="text-[11.5px] font-medium text-ink-0 leading-snug tracking-[-0.005em]"
        style={{ fontVariationSettings: '"wght" 540' }}
      >
        {node.title}
      </div>
    </div>
  );
}
