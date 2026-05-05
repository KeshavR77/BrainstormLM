import { useRef, useCallback, useState, useMemo } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useMotionValueEvent,
  useReducedMotion,
} from 'framer-motion';
import { ArrowRight, Zap, BarChart3, Brain, Sparkles, Shield, Target, Check } from 'lucide-react';
import { ComparisonViewer } from '../components/ComparisonViewer';
import { ScrollTree } from '../components/ScrollTree';

interface Props {
  onLaunchGauntlet: () => void;
  onOpenWorkspace: () => void;
}

interface Particle {
  cx: number; cy: number; r: number; op: number; delay: number; dur: number; hue: 'gold' | 'teal';
}

const HERO_STATS = [
  { value: '-18.4%', label: 'Near-duplicate mass', note: 'Control 0.327 -> 0.267' },
  { value: '+5.7%', label: 'Mean Vendi lift', note: 'Control 1.897 -> 2.004' },
  { value: '+25%', label: 'Pass@5 lift', note: 'Baseline 80% -> 100%' },
  { value: '+18.2%', label: 'Entropy rollup lift', note: '3-prompt avg vs baseline' },
];

function useParticles(count: number, seed = 7): Particle[] {
  return useMemo(() => {
    const rand = (index: number) => {
      const x = Math.sin(seed * 1000 + index * 97.13) * 10000;
      return x - Math.floor(x);
    };
    return Array.from({ length: count }, (_, i) => ({
      cx: rand(i * 7) * 100,
      cy: rand(i * 7 + 1) * 100,
      r: 0.5 + rand(i * 7 + 2) * 1.2,
      op: 0.15 + rand(i * 7 + 3) * 0.35,
      delay: rand(i * 7 + 4) * 14,
      dur: 14 + rand(i * 7 + 5) * 12,
      hue: rand(i * 7 + 6) > 0.78 ? 'teal' as const : 'gold' as const,
    }));
  }, [count, seed]);
}

export function Corpus1Landing({ onLaunchGauntlet, onOpenWorkspace }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const act2Ref = useRef<HTMLDivElement>(null);

  const prefersReduced = useReducedMotion();
  const reduced = !!prefersReduced;
  const particles = useParticles(reduced ? 0 : 46);

  /* ── Scroll tracking ──────────────────────────────────────────────── */
  const { scrollYProgress: pageProg } = useScroll({
    target: rootRef,
    offset: ['start start', 'end end'],
  });

  // Tree section progress
  const { scrollYProgress: a2 } = useScroll({
    target: act2Ref,
    offset: ['start 65%', 'end end'],
  });

  const [treeProg, setTreeProg] = useState(0);
  useMotionValueEvent(a2, 'change', setTreeProg);

  const bgY    = useTransform(pageProg, [0, 1], [0, -140]);
  const constY = useTransform(pageProg, [0, 1], [0, -220]);
  const glowAOp = useTransform(pageProg, [0, 0.5, 1], [0.55, 0.8, 0.3]);
  const glowBOp = useTransform(pageProg, [0, 0.5, 1], [0.3, 0.55, 0.7]);

  const [navSolid, setNavSolid] = useState(false);
  useMotionValueEvent(pageProg, 'change', v => setNavSolid(v > 0.01));

  /* ── Tree captions + CTA ─────────────────────────────────────────── */
  const c1Op = useTransform(a2, [0.02, 0.10, 0.24, 0.30], [0, 1, 1, 0]);
  const c2Op = useTransform(a2, [0.30, 0.38, 0.54, 0.60], [0, 1, 1, 0]);
  const c3Op = useTransform(a2, [0.60, 0.68, 0.86, 0.92], [0, 1, 1, 0]);
  const ctaOp = useTransform(a2, [0.94, 1.0], [0, 1]);
  const ctaY  = useTransform(a2, [0.94, 1.0], [24, 0]);
  const [ctaVisible, setCtaVisible] = useState(false);
  useMotionValueEvent(ctaOp, 'change', v => setCtaVisible(v > 0.1));

  const treeOp = useTransform(a2, [0, 0.08, 0.90, 1.0], [0, 1, 1, 0.2]);
  const treeSc = useTransform(a2, [0, 0.08, 0.94, 1.0], [0.94, 1, 1, 1.04]);

  /* ── Waitlist ─────────────────────────────────────────────────────── */
  const [email, setEmail] = useState('');
  const [waitlistStatus, setWaitlistStatus] = useState<'idle' | 'success'>('idle');

  const handleWaitlist = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setWaitlistStatus('success');
  }, [email]);

  const scrollToWaitlist = useCallback(() => {
    document.getElementById('waitlist')?.scrollIntoView({ behavior: 'smooth' });
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
          <span className="lp__nav-name">Conceptial</span>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button className="lp__nav-btn" onClick={scrollToWaitlist}>
            Join Waitlist
          </button>
          <button className="lp__nav-btn" onClick={onLaunchGauntlet}>
            Open Test <ArrowRight size={13} />
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
              Every headline below is a percent lift against a control or baseline.
          </p>
          <div className="c1__hero-actions">
            <button className="c1__primary-btn" onClick={onLaunchGauntlet}>
              <Zap size={16} /> Open Test
            </button>
            <button className="c1__secondary-btn" onClick={scrollToWaitlist}>
              Join the Waitlist <ArrowRight size={14} />
            </button>
          </div>
        </motion.div>

        <motion.div
          className="c1__hero-stats"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {HERO_STATS.map(stat => (
            <div key={stat.label} className="c1__stat">
              <span className="c1__stat-value">{stat.value}</span>
              <span className="c1__stat-label">{stat.label}</span>
              <span className="c1__stat-note">{stat.note}</span>
            </div>
          ))}
        </motion.div>

        <motion.div
          className="lp__cue"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
        >
          <div className="lp__cue-line" />
          <span>Scroll</span>
        </motion.div>
      </section>

      {/* ═══ ACT 2 — Sticky scroll-driven idea tree ═══ */}
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
              {"Start with any strategic question. Corpius 1 steers it into "}
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

          {/* CTA overlay at end of tree scroll */}
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
              <span className="lp__accent">Corpius 1 explores.</span>
            </h2>
            <p className="lp__final-sub">
              Steerable 14B models. Measured diversity. Business-coherent outliers.
            </p>
            <button className="lp__final-btn" onClick={onLaunchGauntlet}>
              Open Test <ArrowRight size={18} />
            </button>
          </motion.div>
        </div>
      </div>

      {/* ═══ CONTRAST — Comparison Viewer ═══ */}
      <section className="c1__section">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">The Contrast</span>
            <h2 className="landing-section__title">
              Same prompt. <span className="landing-gradient-text">Radically different</span> output.
            </h2>
            <p className="landing-section__sub">
              Standard LLMs converge on the semantic average. Corpius 1 steers the model's internal
              representations to surface ideas that live on the edge of the distribution.
            </p>
          </div>
          <ComparisonViewer />
        </div>
      </section>

      {/* ═══ SCIENCE ═══ */}
      <section className="c1__section c1__section--dark">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">The Science</span>
            <h2 className="landing-section__title">
              Measured divergence, <span className="landing-gradient-text">not random noise.</span>
            </h2>
            <p className="landing-section__sub">
              The headline story is lift against controls: less repetition, higher embedding diversity,
              broader coverage, and faster first tokens.
            </p>
          </div>

          <div className="c1__science-grid">
            {[
              {
                icon: <BarChart3 size={22} />,
                title: '−18.4% near-duplicate rate',
                body: 'Best investor-deck surface: promoted Corpius Arm B reduced near-duplicate mass from 0.3270 control to 0.2667. Lower is better: fewer repeated ideas, less semantic sameness.',
                bar: { width: '82%', cls: 'c1__science-fill--gold', val: '0.2667 vs 0.3270 control', valCls: 'c1__science-val--gold' },
              },
              {
                icon: <Shield size={22} />,
                title: '+5.7% mean Vendi lift',
                body: 'Mean embedding-space diversity rose from 1.8967 control to 2.0042 on the promoted Corpius surface. Vendi is the matrix-entropy diversity score; higher means the set covers more semantic territory.',
                bar: { width: '88%', cls: 'c1__science-fill--teal', val: '2.0042 vs 1.8967 control', valCls: 'c1__science-val--teal' },
              },
              {
                icon: <Brain size={22} />,
                title: '+250% archetypes on the hardest test',
                body: 'On the Legal-Tech GTM prompt, valid strategic archetypes increased from 4 baseline to 14 Corpius, with semantic entropy up +171.1%. This is the single-test highlight, not the all-prompt average.',
                bar: { width: '100%', cls: 'c1__science-fill--gold', val: '4 -> 14 archetypes; H 1.2320 -> 3.3394', valCls: 'c1__science-val--gold' },
              },
            ].map((card, i) => (
              <motion.div
                key={card.title}
                className="c1__science-card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              >
                <div className="feature-card__icon">{card.icon}</div>
                <h3 className="feature-card__title">{card.title}</h3>
                <p className="feature-card__body">{card.body}</p>
                {card.bar && (
                  <div className="c1__science-metric">
                    <div className="c1__science-bar">
                      <div className={`c1__science-fill ${card.bar.cls}`} style={{ width: card.bar.width }} />
                    </div>
                    <span className={`c1__science-val ${card.bar.valCls}`}>{card.bar.val}</span>
                  </div>
                )}
              </motion.div>
            ))}
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
              { num: '01', icon: <Target size={20} />, title: 'Enter a strategic question', desc: 'Any open-ended business, product, or research question. The broader, the better.' },
              { num: '02', icon: <Sparkles size={20} />, title: 'Tune the steering vector', desc: 'Adjust alpha strength and injection layer. Higher alpha = more divergent. Mid-layer = creative. Late-layer = precise.' },
              { num: '03', icon: <Zap size={20} />, title: 'Harvest the outliers', desc: 'The steered 14B model generates ideas that sit on the frontier — novel enough to matter, coherent enough to ship.' },
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

      {/* ═══ WAITLIST ═══ */}
      <section className="c1__section c1__section--dark" id="waitlist">
        <div className="landing-container">
          <motion.div
            className="c1__waitlist"
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.6 }}
          >
            <span className="landing-eyebrow">Early Access</span>
            <h2 className="c1__waitlist-title">
              Be first in when<br />
              <span className="lp__accent">Corpius 1 ships.</span>
            </h2>
            <p className="c1__waitlist-sub">
              We're onboarding a small cohort of founders and innovation leads.
              Drop your email — we'll reach out when the API is ready.
            </p>

            {waitlistStatus === 'success' ? (
              <motion.div
                className="c1__waitlist-success"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <div className="c1__waitlist-success-icon">
                  <Check size={20} />
                </div>
                <div>
                  <p className="c1__waitlist-success-title">You're on the list.</p>
                  <p className="c1__waitlist-success-sub">We'll be in touch when access opens.</p>
                </div>
              </motion.div>
            ) : (
              <form className="c1__waitlist-form" onSubmit={handleWaitlist}>
                <div className="c1__waitlist-input-wrap">
                  <input
                    type="email"
                    placeholder="your@company.com"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    className="c1__waitlist-input"
                    required
                  />
                  <button type="submit" className="c1__waitlist-btn">
                    Join Waitlist <ArrowRight size={15} />
                  </button>
                </div>
                <p className="c1__waitlist-note">No spam. Unsubscribe anytime.</p>
              </form>
            )}

            <div className="c1__waitlist-badges">
              <div className="c1__waitlist-badge">
                <Zap size={12} /> Vendi +5.7% vs control
              </div>
              <div className="c1__waitlist-badge">
                <Shield size={12} /> Pass@5 +25% vs baseline
              </div>
              <div className="c1__waitlist-badge">
                <Brain size={12} /> Near-dup −18.4%
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══ FINAL CTA ═══ */}
      <section className="landing-cta">
        <div className="landing-container">
          <div className="landing-cta__inner" style={{ margin: '0 auto' }}>
            <h2 className="landing-cta__title">
              Standard models converge.<br />
              <span className="landing-gradient-text">Corpius 1 explores.</span>
            </h2>
            <p className="landing-cta__sub">
              Stop getting the same answer as everyone else. Steer a 14B model into
              the idea-space that frontier APIs can't reach.
            </p>
            <button className="landing-cta__button" onClick={onLaunchGauntlet}>
              <Zap size={16} /> Open Test <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="landing-container">
          <div className="landing-footer__inner">
            <span className="landing-footer__brand">Conceptial</span>
            <span className="landing-footer__copy">&copy; 2026</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
