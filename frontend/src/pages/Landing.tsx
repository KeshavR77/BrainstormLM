/**
 * Landing.tsx — Scroll-driven landing page.
 *
 * Refinement pass:
 *   • Scroll-linked background parallax (the fixed bg drifts as you scroll)
 *   • Constellation particle field for ambient depth
 *   • Bridge beam that connects hero input → seed node
 *   • Retuned act-to-act overlap (hero holds longer, tree enters smoother)
 *   • Pin-exit fade so the tree bows out gracefully into the footer
 *   • Respects prefers-reduced-motion
 */

import { useRef, useCallback, useState, useEffect, useMemo } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useMotionValueEvent,
  useReducedMotion,
} from 'framer-motion';
import { Lightbulb, ArrowRight, Zap } from 'lucide-react';
import { ScrollTree } from '../components/ScrollTree';
import { useProjects } from '../hooks/useProjects';

interface LandingProps {
  onStartProject: (projectId: string) => void;
  onOpenDashboard: () => void;
}

/* ── Constellation — drifting particle field ───────────────────────────── */

interface Particle { cx: number; cy: number; r: number; op: number; delay: number; dur: number; hue: 'gold' | 'teal'; }

function useParticles(count: number, seed = 7): Particle[] {
  return useMemo(() => {
    // Deterministic pseudo-random so SSR/CSR match and dev reloads are stable.
    let s = seed;
    const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    return Array.from({ length: count }, () => ({
      cx: rand() * 100,
      cy: rand() * 100,
      r: 0.5 + rand() * 1.2,
      op: 0.15 + rand() * 0.35,
      delay: rand() * 14,
      dur: 14 + rand() * 12,
      hue: rand() > 0.78 ? 'teal' : 'gold',
    }));
  }, [count, seed]);
}

export function Landing({ onStartProject, onOpenDashboard }: LandingProps) {
  const { createProject } = useProjects();
  const [idea, setIdea] = useState('');

  const rootRef = useRef<HTMLDivElement>(null);
  const act1Ref = useRef<HTMLDivElement>(null);
  const act2Ref = useRef<HTMLDivElement>(null);

  const handleSubmit = useCallback(() => {
    const t = idea.trim();
    if (!t) return;
    const p = createProject(t.length > 40 ? t.slice(0, 40) + '…' : t, t);
    onStartProject(p.id);
  }, [idea, createProject, onStartProject]);

  /* ── Scroll tracking ─────────────────────────────────────────────── */

  // Page-level progress for the fixed background parallax.
  const { scrollYProgress: pageProg } = useScroll({
    target: rootRef,
    offset: ['start start', 'end end'],
  });

  const { scrollYProgress: a1 } = useScroll({
    target: act1Ref,
    offset: ['start start', 'end start'],
  });

  // 'start 65%' overlap: tree awakens just as the hero is bowing out.
  // Combined with the retuned hero curve below, this eliminates any
  // black mid-frame while keeping the two acts clearly distinct.
  const { scrollYProgress: a2 } = useScroll({
    target: act2Ref,
    offset: ['start 65%', 'end end'],
  });

  const [treeProg, setTreeProg] = useState(0);
  useMotionValueEvent(a2, 'change', setTreeProg);

  /* ── Act 1 — hero fades a bit slower and translates less ─────────── */
  const heroOp  = useTransform(a1, [0, 0.45], [1, 0]);
  const heroY   = useTransform(a1, [0, 0.5], [0, -32]);
  const inputOp = useTransform(a1, [0.15, 0.55], [1, 0]);
  const inputSc = useTransform(a1, [0.1, 0.55], [1, 0.7]);
  const inputY  = useTransform(a1, [0.1, 0.55], [0, 40]); // settles into seed
  const subOp   = useTransform(a1, [0, 0.12], [1, 0]);
  const bgOp    = useTransform(a1, [0.3, 0.7], [1, 0]);

  /* ── Act 2 — captions retimed so they breathe ────────────────────── */
  const c1Op = useTransform(a2, [0.02, 0.10, 0.24, 0.30], [0, 1, 1, 0]);
  const c2Op = useTransform(a2, [0.30, 0.38, 0.54, 0.60], [0, 1, 1, 0]);
  const c3Op = useTransform(a2, [0.60, 0.68, 0.86, 0.92], [0, 1, 1, 0]);

  /* ── CTA — let the last caption breathe before it arrives ────────── */
  const ctaOp = useTransform(a2, [0.94, 1.0], [0, 1]);
  const ctaY  = useTransform(a2, [0.94, 1.0], [24, 0]);
  const [ctaVisible, setCtaVisible] = useState(false);
  useMotionValueEvent(ctaOp, 'change', v => setCtaVisible(v > 0.1));

  /* ── Tree exits softly as the pin ends ───────────────────────────── */
  const treeOp  = useTransform(a2, [0, 0.08, 0.90, 1.0], [0, 1, 1, 0.2]);
  const treeSc  = useTransform(a2, [0, 0.08, 0.94, 1.0], [0.94, 1, 1, 1.04]);

  /* ── Nav — solidifies once hero begins to leave ──────────────────── */
  const [navSolid, setNavSolid] = useState(false);
  useMotionValueEvent(a1, 'change', v => setNavSolid(v > 0.08));

  /* ── Background parallax (scroll-linked drift on the fixed layer) ── */
  const bgY       = useTransform(pageProg, [0, 1], [0, -140]);
  const constY    = useTransform(pageProg, [0, 1], [0, -220]);
  const glowAOpM  = useTransform(pageProg, [0, 0.5, 1], [0.55, 0.8, 0.3]);
  const glowBOpM  = useTransform(pageProg, [0, 0.5, 1], [0.3, 0.55, 0.7]);

  /* ── Reduced motion ──────────────────────────────────────────────── */
  const prefersReduced = useReducedMotion();
  const [reduced, setReduced] = useState(false);
  useEffect(() => setReduced(!!prefersReduced), [prefersReduced]);

  const particles = useParticles(reduced ? 0 : 46);

  return (
    <div className="lp" ref={rootRef}>
      {/* ═══ Fixed background layers ═══ */}
      <motion.div className="lp__bg" aria-hidden="true" style={{ y: bgY }}>
        <div className="lp__grid" />
        <div className="lp__noise" />
        {!reduced && (
          <>
            <motion.div className="lp__glow lp__glow--a" style={{ opacity: glowAOpM }} />
            <motion.div className="lp__glow lp__glow--b" style={{ opacity: glowBOpM }} />
            <motion.svg
              className="lp__constellation"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ y: constY }}
            >
              {particles.map((p, i) => (
                <circle
                  key={i}
                  cx={p.cx}
                  cy={p.cy}
                  r={p.r * 0.08}
                  fill={p.hue === 'teal' ? '#5eead4' : '#c9a55c'}
                  opacity={p.op}
                  style={{
                    animation: `lpTwinkle ${p.dur}s ease-in-out ${p.delay}s infinite`,
                    transformOrigin: `${p.cx}% ${p.cy}%`,
                  }}
                />
              ))}
            </motion.svg>
          </>
        )}
      </motion.div>

      {/* Nav */}
      <nav className={`lp__nav ${navSolid ? 'lp__nav--solid' : ''}`}>
        <div className="lp__nav-left">
          <div className="lp__nav-mark"><Zap size={13} /></div>
          <span className="lp__nav-name">Conceptial</span>
        </div>
        <button className="lp__nav-btn" onClick={onOpenDashboard}>
          Open Workspace <ArrowRight size={13} />
        </button>
      </nav>

      {/* ═══ ACT 1 — Hero ═══ */}
      <div ref={act1Ref} className="lp__act1">
        <motion.div className="lp__hero" style={{ opacity: bgOp }}>
          <motion.div style={{ opacity: heroOp, y: heroY }}>
            <motion.p className="lp__eyebrow" style={{ opacity: subOp }}>
              Divergence engine
            </motion.p>
            <h1 className="lp__h1">
              Every model gives you<br />
              <span className="lp__accent">the same answer.</span>
            </h1>
          </motion.div>

          <motion.form
            className="lp__form"
            onSubmit={e => { e.preventDefault(); handleSubmit(); }}
            style={{ scale: inputSc, opacity: inputOp, y: inputY }}
          >
            <div className="lp__form-glow" />
            <div className="lp__form-inner">
              <input
                type="text"
                placeholder="How might we differentiate in a crowded market?"
                value={idea}
                onChange={e => setIdea(e.target.value)}
                className="lp__input"
              />
              <button
                type="submit"
                className="lp__submit"
                disabled={!idea.trim()}
                aria-label="Start"
              >
                <Lightbulb size={18} />
              </button>
            </div>
            <motion.p className="lp__sub" style={{ opacity: subOp }}>
              Type a strategic question, or scroll to see it work
            </motion.p>
          </motion.form>

          <motion.div className="lp__cue" style={{ opacity: subOp }}>
            <div className="lp__cue-line" />
            <span>Scroll</span>
          </motion.div>
        </motion.div>
      </div>

      {/* ═══ ACT 2 — Sticky tree + captions + CTA ═══ */}
      <div ref={act2Ref} className="lp__act2">
        <div className="lp__pin">
          <motion.div
            className="lp__tree-area"
            style={{ opacity: treeOp, scale: treeSc }}
          >
            <ScrollTree progress={treeProg} className="lp__tree" />
          </motion.div>

          <div className="lp__captions">
            <motion.p className="lp__cap" style={{ opacity: c1Op }}>
              {"Start with any strategic question. Conceptial steers it into "}
              <strong>{"genuinely different directions"}</strong>{" — not variations of the same answer."}
            </motion.p>
            <motion.p className="lp__cap" style={{ opacity: c2Op }}>
              {"Choose the direction that surprises you — it becomes "}
              <strong>{"the new seed"}</strong>{" and branches deeper."}
            </motion.p>
            <motion.p className="lp__cap" style={{ opacity: c3Op }}>
              {"One question becomes a "}
              <strong>{"whole landscape of divergent possibilities"}</strong>{" — each fork further from the last."}
            </motion.p>
          </div>

          {/* CTA overlay */}
          <motion.div
            className="lp__cta-overlay"
            style={{
              opacity: ctaOp,
              y: ctaY,
              pointerEvents: ctaVisible ? 'auto' : 'none',
            }}
          >
            <h2 className="lp__final-title">
              Frontier models converge.<br />
              <span className="lp__accent">Conceptial explores.</span>
            </h2>
            <p className="lp__final-sub">
              Turn one question into a tree of meaningfully different ideas.
            </p>
            <button className="lp__final-btn" onClick={onOpenDashboard}>
              Open workspace <ArrowRight size={18} />
            </button>
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <footer className="lp__footer">
        <span className="lp__footer-name">Conceptial</span>
        <span className="lp__footer-copy">© 2026</span>
      </footer>
    </div>
  );
}
