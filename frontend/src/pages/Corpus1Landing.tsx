import { useRef, useCallback, useState, useMemo } from 'react';
import {
  motion,
  useScroll,
  useTransform,
  useMotionValueEvent,
  useReducedMotion,
} from 'framer-motion';
import { ArrowRight, Zap, BarChart3, Brain, Shield, Check } from 'lucide-react';
import { ComparisonViewer } from '../components/ComparisonViewer';

interface Props {
  onLaunchGauntlet: () => void;
  onOpenWorkspace: () => void;
}

interface Particle {
  cx: number; cy: number; r: number; op: number; delay: number; dur: number; hue: 'gold' | 'teal';
}

const HERO_STATS = [
  { value: '-18%', label: 'Fewer repeat answers', note: 'Near-duplicates vs control' },
  { value: '+11%', label: 'Answer diversity lift', note: 'Mean diversity score vs control' },
  { value: '+25%', label: 'More correct paths', note: 'Solution coverage lift' },
  { value: '100%', label: 'Quality gate pass', note: 'Coherence + relevance held' },
];

const DESIGN_TREE = {
  prompt: 'How should a design team redesign onboarding so new users reach an aha moment in under 60 seconds?',
  branches: [
    { title: 'Instant personal payoff', note: 'Start from the user\'s goal, not the product tour' },
    { title: 'Progressive trust', note: 'Prove value before asking for setup work' },
    { title: 'Social proof moment', note: 'Show what similar teams achieved first' },
  ],
  deep: [
    { title: 'Role-based first task', note: 'Designer, founder, or PM each gets a different first win' },
    { title: 'Sample data sandbox', note: 'Let users feel the finished workflow before connecting tools' },
    { title: 'Outcome receipt', note: 'End onboarding with a concrete artifact they can share' },
  ],
};

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

  const prefersReduced = useReducedMotion();
  const reduced = !!prefersReduced;
  const particles = useParticles(reduced ? 0 : 46);

  /* ── Scroll tracking ──────────────────────────────────────────────── */
  const { scrollYProgress: pageProg } = useScroll({
    target: rootRef,
    offset: ['start start', 'end end'],
  });

  const bgY    = useTransform(pageProg, [0, 1], [0, -140]);
  const constY = useTransform(pageProg, [0, 1], [0, -220]);
  const glowAOp = useTransform(pageProg, [0, 0.5, 1], [0.55, 0.8, 0.3]);
  const glowBOp = useTransform(pageProg, [0, 0.5, 1], [0.3, 0.55, 0.7]);

  const [navSolid, setNavSolid] = useState(false);
  useMotionValueEvent(pageProg, 'change', v => setNavSolid(v > 0.01));

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

      {/* ═══ SCIENCE ═══ */}
      <section className="c1__section c1__section--dark">
        <div className="landing-container">
          <div className="landing-section__header">
            <span className="landing-eyebrow">The Science</span>
            <h2 className="landing-section__title">
              More diversity, <span className="landing-gradient-text">without quality drop.</span>
            </h2>
            <p className="landing-section__sub">
              The benchmark checks both sides: outputs should cover more territory and still pass
              coherence and relevance checks.
            </p>
          </div>

          <div className="c1__science-grid">
            {[
              {
                icon: <BarChart3 size={22} />,
                title: '−18% near-duplicate rate',
                body: 'Best investor-deck surface: promoted Corpius Arm B reduced near-duplicate mass from 0.33 control to 0.27. Lower is better: fewer repeated ideas, less semantic sameness.',
                bar: { width: '82%', cls: 'c1__science-fill--gold', val: '0.27 vs 0.33 control', valCls: 'c1__science-val--gold' },
              },
              {
                icon: <Shield size={22} />,
                title: '+11% answer diversity lift',
                body: 'Best mean Vendi lift found in the local v17 PRISM trail: qwen_v17 rose from 1.17 OpenAI control to 1.29. The stricter investor-deck matched-control lift is +6%.',
                bar: { width: '88%', cls: 'c1__science-fill--teal', val: '1.29 vs 1.17 local control', valCls: 'c1__science-val--teal' },
              },
              {
                icon: <Brain size={22} />,
                title: '+25% more correct paths',
                body: 'In the benchmark, the model gets five attempts and needs at least one answer that solves the task. Corpius improved that coverage by 25% relative to baseline.',
                bar: { width: '100%', cls: 'c1__science-fill--gold', val: 'More coverage with the same 5-attempt budget', valCls: 'c1__science-val--gold' },
              },
              {
                icon: <Check size={22} />,
                title: '100% quality gate pass',
                body: 'The added diversity did not come from junk output. Corpius passed the coherence and relevance quality floor on every judged sample, ahead of the unsteered baseline.',
                bar: { width: '100%', cls: 'c1__science-fill--teal', val: '100% vs 98% unsteered', valCls: 'c1__science-val--teal' },
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

      {/* ═══ TREE EXAMPLE ═══ */}
      <section className="c1__section c1__tree-section">
        <div className="landing-container">
          <div className="c1__tree-demo">
            <div className="c1__tree-copy">
              <span className="landing-eyebrow">Idea Map</span>
              <h2 className="landing-section__title">
                A designer's question becomes a <span className="landing-gradient-text">usable map.</span>
              </h2>
              <p className="landing-section__sub">
                The test is preloaded with a product-design prompt so the room sees branching,
                selection, and deeper exploration immediately.
              </p>
              <button className="c1__secondary-btn" onClick={onLaunchGauntlet}>
                Open Test Map <ArrowRight size={14} />
              </button>
            </div>

            <motion.div
              className="c1__tree-canvas"
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.55 }}
            >
              <div className="c1__tree-root">
                <span>Designer prompt</span>
                <strong>{DESIGN_TREE.prompt}</strong>
              </div>

              <div className="c1__tree-branches">
                {DESIGN_TREE.branches.map((branch, idx) => (
                  <article
                    key={branch.title}
                    className={`c1__tree-node ${idx === 0 ? 'c1__tree-node--selected' : ''}`}
                  >
                    <span className="c1__tree-node-kicker">{idx === 0 ? 'Selected' : `Option ${idx + 1}`}</span>
                    <h3>{branch.title}</h3>
                    <p>{branch.note}</p>
                  </article>
                ))}
              </div>

              <div className="c1__tree-deep">
                {DESIGN_TREE.deep.map(node => (
                  <article key={node.title} className="c1__tree-node c1__tree-node--deep">
                    <span className="c1__tree-node-kicker">Next move</span>
                    <h3>{node.title}</h3>
                    <p>{node.note}</p>
                  </article>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
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
              Standard LLMs converge on the semantic average. Corpius 1 steers the model's internal
              representations to surface ideas that live on the edge of the distribution.
            </p>
          </div>
          <ComparisonViewer />
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
                <Zap size={12} /> Diversity +11%
              </div>
              <div className="c1__waitlist-badge">
                <Shield size={12} /> Quality gate 100%
              </div>
              <div className="c1__waitlist-badge">
                <Brain size={12} /> Near-dup −18%
              </div>
            </div>
          </motion.div>
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
