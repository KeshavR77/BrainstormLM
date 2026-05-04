import { useRef, useCallback, useState, useEffect, useMemo } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useMotionValueEvent,
  useReducedMotion,
} from 'framer-motion';
import { ArrowRight, Zap, BarChart3, Brain, Sparkles, Shield, Target } from 'lucide-react';
import { ComparisonViewer } from '../components/ComparisonViewer';

interface Props {
  onLaunchGauntlet: () => void;
  onOpenWorkspace: () => void;
}

interface Particle {
  cx: number; cy: number; r: number; op: number; delay: number; dur: number; hue: 'gold' | 'teal';
}

function useParticles(count: number, seed = 7): Particle[] {
  return useMemo(() => {
    let s = seed;
    const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    return Array.from({ length: count }, () => ({
      cx: rand() * 100, cy: rand() * 100, r: 0.5 + rand() * 1.2,
      op: 0.15 + rand() * 0.35, delay: rand() * 14, dur: 14 + rand() * 12,
      hue: rand() > 0.78 ? 'teal' as const : 'gold' as const,
    }));
  }, [count, seed]);
}

export function Corpus1Landing({ onLaunchGauntlet, onOpenWorkspace }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const prefersReduced = useReducedMotion();
  const [reduced, setReduced] = useState(false);
  useEffect(() => setReduced(!!prefersReduced), [prefersReduced]);
  const particles = useParticles(reduced ? 0 : 46);

  const { scrollYProgress: pageProg } = useScroll({
    target: rootRef,
    offset: ['start start', 'end end'],
  });

  const bgY = useTransform(pageProg, [0, 1], [0, -140]);
  const constY = useTransform(pageProg, [0, 1], [0, -220]);
  const glowAOp = useTransform(pageProg, [0, 0.5, 1], [0.55, 0.8, 0.3]);
  const glowBOp = useTransform(pageProg, [0, 0.5, 1], [0.3, 0.55, 0.7]);

  const [navSolid, setNavSolid] = useState(false);
  useMotionValueEvent(pageProg, 'change', v => setNavSolid(v > 0.02));

  const scrollToGauntlet = useCallback(() => {
    document.getElementById('gauntlet-cta')?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return (
    <div className="c1" ref={rootRef}>
      {/* Fixed background */}
      <motion.div className="lp__bg" aria-hidden="true" style={{ y: bgY }}>
        <div className="lp__grid" />
        <div className="lp__noise" />
        {!reduced && (
          <>
            <motion.div className="lp__glow lp__glow--a" style={{ opacity: glowAOp }} />
            <motion.div className="lp__glow lp__glow--b" style={{ opacity: glowBOp }} />
            <motion.svg
              className="lp__constellation"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ y: constY }}
            >
              {particles.map((p, i) => (
                <circle
                  key={i} cx={p.cx} cy={p.cy} r={p.r * 0.08}
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
          <span className="lp__nav-name">Corpus 1</span>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button className="lp__nav-btn" onClick={scrollToGauntlet}>
            Launch Gauntlet <ArrowRight size={13} />
          </button>
          <button className="lp__nav-btn" onClick={onOpenWorkspace}>
            Workspace <ArrowRight size={13} />
          </button>
        </div>
      </nav>

      {/* ═══ HERO ═══ */}
      <section className="c1__hero">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 1, 0.5, 1] }}
        >
          <p className="lp__eyebrow">Non-Linear Reasoning Engine</p>
          <h1 className="c1__h1">
            Break the<br />
            <span className="lp__accent">Echo Chamber.</span>
          </h1>
          <p className="c1__sub">
            Steerable 14B models that bypass the semantic average to find the outliers.
            Representation engineering meets divergent ideation.
          </p>
          <div className="c1__hero-actions">
            <button className="c1__primary-btn" onClick={onLaunchGauntlet}>
              <Zap size={16} /> Launch Gauntlet
            </button>
            <button className="c1__secondary-btn" onClick={scrollToGauntlet}>
              See it work <ArrowRight size={14} />
            </button>
          </div>
        </motion.div>

        <motion.div
          className="c1__hero-stats"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="c1__stat">
            <span className="c1__stat-value">4.72</span>
            <span className="c1__stat-label">Vendi Score</span>
          </div>
          <div className="c1__stat-divider" />
          <div className="c1__stat">
            <span className="c1__stat-value">81%</span>
            <span className="c1__stat-label">PRISM Coherence</span>
          </div>
          <div className="c1__stat-divider" />
          <div className="c1__stat">
            <span className="c1__stat-value">14B</span>
            <span className="c1__stat-label">Parameters</span>
          </div>
        </motion.div>
      </section>

      {/* ═══ CONTRAST — Comparison Viewer ═══ */}
      <section className="c1__section">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">The Contrast</span>
            <h2 className="landing-section__title">
              Same prompt. <span className="landing-gradient-text">Radically different</span> output.
            </h2>
            <p className="landing-section__sub">
              Standard LLMs converge on the semantic average. Corpus 1 steers the model's internal
              representations to surface ideas that live on the edge of the distribution.
            </p>
          </div>
          <ComparisonViewer />
        </div>
      </section>

      {/* ═══ SCIENCE — Vendi Score & PRISM ═══ */}
      <section className="c1__section c1__section--dark">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">The Science</span>
            <h2 className="landing-section__title">
              Measured divergence, <span className="landing-gradient-text">not random noise.</span>
            </h2>
            <p className="landing-section__sub">
              Two metrics ensure steered outputs are genuinely novel and still business-coherent.
            </p>
          </div>

          <div className="c1__science-grid">
            <motion.div
              className="c1__science-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5 }}
            >
              <div className="feature-card__icon"><BarChart3 size={22} /></div>
              <h3 className="feature-card__title">Vendi Score</h3>
              <p className="feature-card__body">
                Measures effective diversity across a batch of generations using matrix-based
                entropy. A score of 4.72 means the outputs span nearly 5 orthogonal semantic
                directions — standard LLMs typically score below 2.0.
              </p>
              <div className="c1__science-metric">
                <div className="c1__science-bar">
                  <div className="c1__science-fill c1__science-fill--gold" style={{ width: '94%' }} />
                </div>
                <span className="c1__science-val c1__science-val--gold">4.72 / 5.0</span>
              </div>
            </motion.div>

            <motion.div
              className="c1__science-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="feature-card__icon"><Shield size={22} /></div>
              <h3 className="feature-card__title">PRISM Coherence</h3>
              <p className="feature-card__body">
                Ensures steered outputs remain actionable and relevant. PRISM scores each idea
                against the original prompt for business logic, feasibility, and specificity.
                Above 75% means the idea is wild but grounded.
              </p>
              <div className="c1__science-metric">
                <div className="c1__science-bar">
                  <div className="c1__science-fill c1__science-fill--teal" style={{ width: '81%' }} />
                </div>
                <span className="c1__science-val c1__science-val--teal">81%</span>
              </div>
            </motion.div>

            <motion.div
              className="c1__science-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="feature-card__icon"><Brain size={22} /></div>
              <h3 className="feature-card__title">Representation Engineering</h3>
              <p className="feature-card__body">
                We don't prompt-hack. We inject learned steering vectors directly into the
                model's transformer layers — nudging the latent space toward unexplored regions
                without destroying coherence.
              </p>
              <div className="c1__science-layers">
                <div className="c1__layer">
                  <span className="c1__layer-name">Layer 14 (Mid)</span>
                  <span className="c1__layer-desc">Creative divergence</span>
                </div>
                <div className="c1__layer">
                  <span className="c1__layer-name">Layer 18 (Late)</span>
                  <span className="c1__layer-desc">Technical precision</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ═══ HOW IT WORKS ═══ */}
      <section className="c1__section">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">How It Works</span>
            <h2 className="landing-section__title">
              From question to <span className="landing-gradient-text">idea-edge</span> in three steps.
            </h2>
          </div>

          <div className="c1__steps">
            {[
              {
                num: '01',
                icon: <Target size={20} />,
                title: 'Enter a strategic question',
                desc: 'Any open-ended business, product, or research question. The broader, the better.'
              },
              {
                num: '02',
                icon: <Sparkles size={20} />,
                title: 'Tune the steering vector',
                desc: 'Adjust alpha strength and injection layer. Higher alpha = more divergent. Mid-layer = creative. Late-layer = precise.'
              },
              {
                num: '03',
                icon: <Zap size={20} />,
                title: 'Harvest the outliers',
                desc: 'The steered 14B model generates ideas that sit on the frontier — novel enough to matter, coherent enough to ship.'
              },
            ].map((step, i) => (
              <motion.div
                key={step.num}
                className="c1__step"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.4, delay: i * 0.12 }}
              >
                <div className="c1__step-num">{step.num}</div>
                <div className="c1__step-icon">{step.icon}</div>
                <h3 className="c1__step-title">{step.title}</h3>
                <p className="c1__step-desc">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ CTA ═══ */}
      <section className="landing-cta" id="gauntlet-cta">
        <div className="landing-container">
          <div className="landing-cta__inner" style={{ margin: '0 auto' }}>
            <h2 className="landing-cta__title">
              Standard models converge.<br />
              <span className="landing-gradient-text">Corpus 1 explores.</span>
            </h2>
            <p className="landing-cta__sub">
              Stop getting the same answer as everyone else. Steer a 14B model into
              the idea-space that frontier APIs can't reach.
            </p>
            <button className="landing-cta__button" onClick={onLaunchGauntlet}>
              <Zap size={16} /> Launch Gauntlet <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="landing-container">
          <div className="landing-footer__inner">
            <span className="landing-footer__brand">Corpus 1</span>
            <span className="landing-footer__copy">&copy; 2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
