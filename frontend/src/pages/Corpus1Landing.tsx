/**
 * Corpus1Landing — Conceptial marketing surface.
 *
 * Pollen × Prism palette. Editorial Fraunces display + Inter Variable body.
 * Sections (top → bottom):
 *   1. Hero — wordmark, headline, CTAs, animated idea-tree centerpiece.
 *   2. Principles — three editorial lines.
 *   3. Comparison scrubber — linear vs Conceptial split, scroll-driven.
 *   4. Science strip — compressed metric chips.
 *   5. Waitlist — minimal email form.
 *   6. Footer — wordmark + minimal links.
 */

import { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ArrowUpRight, Check } from 'lucide-react';
import { CursorSpotlight } from '../components/landing/CursorSpotlight';
import { MorphingWordmark } from '../components/landing/MorphingWordmark';
import { HeroTreeGrowth } from '../components/landing/HeroTreeGrowth';
import { ScrollScrubber } from '../components/landing/ScrollScrubber';
import { Wordmark } from '../components/Logo';
import { fadeRise, linesReveal, popIn, staggerContainer, staggerFast, depthHue as depthHueAt } from '../styles/motion';

interface Props {
  onLaunchGauntlet: () => void;
  onLaunchDemo: () => void;
  onOpenWorkspace: () => void;
}

const PRINCIPLES = [
  'The first answer is rarely the best answer.',
  'Diversity over polish. Coverage over consensus.',
  'A search through ideas, not a vote between them.',
];

const METRICS = [
  { value: '−18%',  label: 'Fewer repeat answers',   note: 'Near-duplicate rate 0.27 vs 0.33 baseline',     tone: 'gen' as const, hue: 320 },
  { value: '+11%',  label: 'More answer diversity',  note: 'Vendi score 1.29 vs 1.17 baseline',             tone: 'gen' as const, hue: 280 },
  { value: '+25%',  label: 'More correct paths',     note: 'Pass@5 — 1+ valid answer in 5 attempts',        tone: 'gen' as const, hue: 25  },
  { value: '100%',  label: 'Quality gate pass',      note: 'Coherence + relevance, every judged sample',    tone: 'brand' as const, hue: 0  },
];

export function Corpus1Landing({ onLaunchGauntlet, onLaunchDemo, onOpenWorkspace }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [email, setEmail] = useState('');
  const [waitlistOk, setWaitlistOk] = useState(false);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    setWaitlistOk(true);
  }, [email]);

  return (
    <div ref={rootRef} className="relative min-h-screen bg-paper text-ink-0 font-sans antialiased overflow-x-hidden">
      {/* Cursor spotlight backdrop */}
      <CursorSpotlight />

      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-40 backdrop-blur-md bg-paper/75 border-b border-ink-line">
        <div className="max-w-7xl mx-auto px-6 sm:px-10 py-3.5 flex items-center justify-between">
          <a href="#/" className="group">
            <Wordmark size={20} weight={560} className="group-hover:opacity-80 transition-opacity" />
          </a>
          <div className="flex items-center gap-1 sm:gap-2">
            <a
              href="#waitlist"
              className="hidden sm:inline-block text-ink-1 hover:text-ink-0 text-sm px-3 py-1.5 rounded-md transition-colors"
            >
              Waitlist
            </a>
            <button
              onClick={onLaunchDemo}
              className="text-ink-1 hover:text-ink-0 text-sm px-3 py-1.5 rounded-md transition-colors hidden sm:inline-flex items-center gap-1"
            >
              Demo <ArrowUpRight size={13} />
            </button>
            <button
              onClick={onLaunchGauntlet}
              className="text-ink-1 hover:text-ink-0 text-sm px-3 py-1.5 rounded-md transition-colors hidden sm:inline-flex items-center gap-1"
            >
              Open Test <ArrowUpRight size={13} />
            </button>
            <button
              onClick={onOpenWorkspace}
              className="bg-ink-0 text-paper text-sm px-4 py-1.5 rounded-full font-medium hover:bg-ink-1 transition-colors inline-flex items-center gap-1.5"
            >
              Workspace <ArrowRight size={13} />
            </button>
          </div>
        </div>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <section className="relative pt-28 pb-20 sm:pt-32 sm:pb-28 px-6 sm:px-10">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-center">
          {/* Left: copy */}
          <motion.div
            className="lg:col-span-6"
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
          >
            <motion.div
              variants={fadeRise}
              className="inline-flex items-center gap-2.5 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-6"
            >
              <span className="h-px w-6 bg-ink-3" />
              Non-linear reasoning engine
            </motion.div>

            <div className="overflow-hidden pb-1">
              <motion.h1
                variants={linesReveal}
                className="font-display text-ink-0 leading-[0.95] tracking-[-0.025em] text-[clamp(52px,8.5vw,120px)]"
                style={{ fontVariationSettings: '"wght" 360, "opsz" 144' }}
              >
                Think
              </motion.h1>
            </div>
            <div className="overflow-hidden pb-1 -mt-1">
              <motion.h1
                variants={linesReveal}
                className="font-display text-ink-0 leading-[0.95] tracking-[-0.025em] text-[clamp(52px,8.5vw,120px)] italic"
                style={{ fontVariationSettings: '"wght" 360, "opsz" 144, "SOFT" 100' }}
              >
                in branches.
              </motion.h1>
            </div>

            <motion.p
              variants={fadeRise}
              className="mt-7 text-ink-1 text-[17px] sm:text-[19px] leading-[1.55] max-w-[500px]"
            >
              A brainstorm tool that returns many genuinely different directions
              instead of one polished answer. Built on representation-engineered
              models that bypass the semantic average.
            </motion.p>

            <motion.div variants={fadeRise} className="mt-9 flex items-center gap-3">
              <button
                onClick={onLaunchGauntlet}
                className="inline-flex items-center gap-2 bg-brand hover:bg-brand-deep text-white text-[14.5px] font-medium px-6 py-3.5 rounded-full shadow-brand transition-all duration-200 hover:translate-y-[-1px]"
              >
                Try the test
                <ArrowRight size={16} />
              </button>
              <a
                href="#waitlist"
                className="inline-flex items-center gap-2 text-ink-0 text-[14.5px] font-medium px-6 py-3.5 rounded-full border border-[var(--ink-line-bold)] hover:bg-paper-2 transition-all duration-200"
              >
                Join waitlist
              </a>
            </motion.div>

            <motion.div
              variants={fadeRise}
              className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11.5px] text-ink-2"
            >
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1 w-1 rounded-full bg-brand" />
                −18% repeat answers
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1 w-1 rounded-full" style={{ background: 'oklch(0.62 0.20 140)' }} />
                +11% answer diversity
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1 w-1 rounded-full" style={{ background: 'oklch(0.62 0.20 200)' }} />
                +25% more correct paths
              </span>
            </motion.div>
          </motion.div>

          {/* Right: animated idea tree (chrome lives inside the component) */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.95, ease: [0.22, 1, 0.36, 1], delay: 0.25 }}
            className="lg:col-span-6"
          >
            <div className="relative aspect-[4/3] rounded-2xl bg-paper-1 border border-ink-line overflow-hidden shadow-lift">
              <HeroTreeGrowth className="px-6 py-5 sm:px-8 sm:py-7" />
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── PRINCIPLES ───────────────────────────────────────────────── */}
      <section className="relative py-24 sm:py-32 px-6 sm:px-10 border-t border-ink-line">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="inline-flex items-center gap-2.5 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-12"
          >
            <span className="h-px w-6 bg-ink-3" />
            Principles
          </motion.div>
          <ol className="space-y-6 sm:space-y-9">
            {PRINCIPLES.map((p, i) => (
              <li key={p} className="overflow-hidden">
                <motion.div
                  initial={{ y: 60, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.95, ease: [0.22, 1, 0.36, 1], delay: i * 0.12 }}
                  className="font-display text-ink-0 text-[clamp(28px,4.6vw,60px)] leading-[1.06] tracking-[-0.018em] flex items-baseline gap-5"
                  style={{ fontVariationSettings: `"wght" ${360 + i * 40}, "opsz" ${72 + i * 20}` }}
                >
                  <span
                    className="font-mono tabular-nums text-[0.32em] text-ink-3 tracking-[0.1em] pt-2"
                    style={{ fontVariationSettings: 'normal' }}
                  >
                    {`0${i + 1} —`}
                  </span>
                  <span>{p}</span>
                </motion.div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── COMPARISON SCRUBBER ──────────────────────────────────────── */}
      <section className="relative py-24 sm:py-28 px-6 sm:px-10 border-t border-ink-line bg-paper-1">
        <div className="max-w-7xl mx-auto">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            className="text-center mb-14"
          >
            <motion.div variants={fadeRise} className="inline-flex items-center gap-2.5 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-6">
              <span className="h-px w-6 bg-ink-3" />
              Same prompt, two engines
              <span className="h-px w-6 bg-ink-3" />
            </motion.div>
            <div className="overflow-hidden">
              <motion.h2
                variants={linesReveal}
                className="font-display text-ink-0 text-[clamp(32px,5vw,60px)] leading-[1.05] tracking-[-0.018em] max-w-3xl mx-auto"
                style={{ fontVariationSettings: '"wght" 400, "opsz" 96' }}
              >
                Frontier models converge.
              </motion.h2>
            </div>
            <div className="overflow-hidden">
              <motion.h2
                variants={linesReveal}
                className="font-display italic text-ink-2 text-[clamp(32px,5vw,60px)] leading-[1.05] tracking-[-0.018em] max-w-3xl mx-auto"
                style={{ fontVariationSettings: '"wght" 380, "opsz" 96, "SOFT" 100' }}
              >
                Conceptial diverges.
              </motion.h2>
            </div>
          </motion.div>
          <ScrollScrubber />
        </div>
      </section>

      {/* ── HOW IT WORKS ─────────────────────────────────────────────── */}
      <section className="relative py-24 sm:py-28 px-6 sm:px-10 border-t border-ink-line">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-16 items-start">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-100px' }}
              transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
              className="lg:col-span-4 lg:sticky lg:top-32"
            >
              <div className="inline-flex items-center gap-2.5 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-5">
                <span className="h-px w-6 bg-ink-3" />
                How it works
              </div>
              <h2
                className="font-display text-ink-0 text-[clamp(28px,4vw,48px)] leading-[1.05] tracking-[-0.018em]"
                style={{ fontVariationSettings: '"wght" 380, "opsz" 96' }}
              >
                Three moves <span className="italic">between</span> the prompt and the answer.
              </h2>
            </motion.div>
            <ol className="lg:col-span-8 space-y-5">
              {[
                {
                  num: '01',
                  title: 'Steer the activation space',
                  body: 'A purified creative vector pushes the model into less-traveled regions of the residual stream — early in generation, before the answer collapses.',
                  hue: depthHueAt(0),
                },
                {
                  num: '02',
                  title: 'Sample many candidates in parallel',
                  body: 'Instead of one polished answer, we draw 6–14 distinct samples, each anchored to a different archetype of the solution space.',
                  hue: depthHueAt(1),
                },
                {
                  num: '03',
                  title: 'Score, prune, and present',
                  body: 'A quality gate filters incoherent or off-prompt outputs. The remaining tree gets ranked by diversity and surfaced as an interactive map.',
                  hue: depthHueAt(2),
                },
              ].map(({ num, title, body, hue }, i) => (
                <motion.li
                  key={num}
                  initial={{ opacity: 0, y: 32 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-80px' }}
                  transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1], delay: i * 0.12 }}
                  className="grid grid-cols-[auto_1fr] gap-5 items-start p-6 sm:p-7 rounded-2xl border border-ink-line bg-paper hover:bg-paper-1 hover:translate-x-1 transition-all duration-300"
                  style={{ '--gen-hue': hue } as React.CSSProperties}
                >
                  <span
                    className="font-display tabular-nums text-[clamp(28px,3vw,40px)] leading-none tracking-tight"
                    style={{
                      color: 'oklch(0.55 0.18 var(--gen-hue))',
                      fontVariationSettings: '"wght" 380, "opsz" 80',
                    }}
                  >
                    {num}
                  </span>
                  <div>
                    <h3
                      className="font-display text-ink-0 text-[18px] sm:text-[20px] tracking-tight mb-1.5"
                      style={{ fontVariationSettings: '"wght" 520, "opsz" 32' }}
                    >
                      {title}
                    </h3>
                    <p className="text-ink-1 text-[14px] leading-relaxed">{body}</p>
                  </div>
                </motion.li>
              ))}
            </ol>
          </div>
        </div>
      </section>

      {/* ── SCIENCE STRIP ────────────────────────────────────────────── */}
      <section className="relative py-24 sm:py-28 px-6 sm:px-10 border-t border-ink-line">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="mb-10"
          >
            <div className="inline-flex items-center gap-2.5 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-3">
              <span className="h-px w-6 bg-ink-3" />
              The numbers
            </div>
            <h3
              className="font-display text-ink-0 text-[clamp(22px,3vw,32px)] tracking-[-0.012em] max-w-2xl"
              style={{ fontVariationSettings: '"wght" 440, "opsz" 48' }}
            >
              Conceptial vs the same model unsteered, on a repeated-prompt stress test.
            </h3>
          </motion.div>

          <motion.div
            variants={staggerFast}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-60px' }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4"
          >
            {METRICS.map((m) => (
              <motion.div
                key={m.label}
                variants={popIn}
                className="rounded-xl border border-ink-line bg-paper-1 p-5 sm:p-6 hover:bg-paper-2 hover:border-[var(--ink-line-bold)] transition-all duration-300"
                style={{ '--gen-hue': m.hue } as React.CSSProperties}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ background: m.tone === 'brand' ? 'var(--brand)' : 'var(--gen)' }}
                  />
                  <span className="text-ink-2 text-[10.5px] uppercase tracking-[0.18em]">{m.label}</span>
                </div>
                <div
                  className="font-display tabular-nums text-[clamp(36px,5vw,56px)] leading-none tracking-[-0.025em] text-ink-0"
                  style={{ fontVariationSettings: '"wght" 480, "opsz" 96' }}
                >
                  {m.value}
                </div>
                <div className="mt-2 text-ink-3 text-[11px] leading-snug">
                  {m.note}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── WAITLIST ─────────────────────────────────────────────────── */}
      <section id="waitlist" className="relative py-28 sm:py-36 px-6 sm:px-10 border-t border-ink-line bg-paper-2 overflow-hidden">
        {/* Decorative side glyphs */}
        <div aria-hidden className="absolute inset-y-0 left-0 w-1/2 pointer-events-none opacity-40">
          <div className="absolute top-1/2 -translate-y-1/2 left-[-10%] w-[60%] aspect-square rounded-full"
               style={{ background: 'radial-gradient(circle, oklch(0.88 0.10 320 / 0.5), transparent 60%)' }} />
        </div>
        <div aria-hidden className="absolute inset-y-0 right-0 w-1/2 pointer-events-none opacity-40">
          <div className="absolute top-1/2 -translate-y-1/2 right-[-10%] w-[60%] aspect-square rounded-full"
               style={{ background: 'radial-gradient(circle, oklch(0.88 0.10 140 / 0.5), transparent 60%)' }} />
        </div>

        <div className="relative max-w-2xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-6">
            <span className="h-px w-5 bg-ink-3" />
            Early access
            <span className="h-px w-5 bg-ink-3" />
          </div>
          <h3
            className="font-display text-ink-0 text-[clamp(34px,5.5vw,64px)] leading-[1.02] tracking-[-0.02em] mb-5"
            style={{ fontVariationSettings: '"wght" 380, "opsz" 144' }}
          >
            Get on the list.
          </h3>
          <p className="text-ink-1 text-[16px] leading-relaxed mb-10 max-w-md mx-auto">
            Rolling out access to a small batch of design and strategy teams.
            Two lines on what you'd use it for is plenty.
          </p>

          {!waitlistOk ? (
            <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto">
              <input
                type="email"
                required
                placeholder="you@studio.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="flex-1 px-5 py-3.5 rounded-full bg-paper border border-ink-line text-ink-0 placeholder:text-ink-3 outline-none focus:border-brand focus:ring-2 focus:ring-brand/20 transition-all"
              />
              <button
                type="submit"
                className="bg-brand hover:bg-brand-deep text-white text-[14.5px] font-medium px-7 py-3.5 rounded-full shadow-brand transition-all hover:translate-y-[-1px]"
              >
                Request access
              </button>
            </form>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-full bg-brand/10 text-brand border border-brand/30"
            >
              <Check size={16} /> You're on the list.
            </motion.div>
          )}

          {/* Social proof row */}
          <div className="mt-10 flex flex-col items-center gap-4">
            <div className="flex -space-x-2">
              {['#E11D74', 'oklch(0.62 0.20 280)', 'oklch(0.62 0.20 140)', 'oklch(0.62 0.20 25)', 'oklch(0.62 0.20 200)'].map((bg, i) => (
                <span
                  key={i}
                  className="h-7 w-7 rounded-full border-2 border-paper-2 flex items-center justify-center text-[10px] text-white font-medium"
                  style={{ background: bg }}
                >
                  {['M', 'D', 'L', 'A', 'R'][i]}
                </span>
              ))}
            </div>
            <p className="text-ink-2 text-[12px]">
              Joining design teams at <span className="text-ink-1">studios</span>, <span className="text-ink-1">consultancies</span>, and <span className="text-ink-1">research labs</span>.
            </p>
          </div>

          <p className="mt-6 text-ink-3 text-[11.5px]">No spam · unsubscribe anytime</p>
        </div>
      </section>

      {/* ── FOOTER ───────────────────────────────────────────────────── */}
      <footer className="relative px-6 sm:px-10 pt-20 pb-10 border-t border-ink-line bg-paper">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-14 items-start">
            <div className="md:col-span-2">
              <Wordmark size={22} weight={560} />
              <p className="mt-3 text-ink-2 text-[13px] max-w-md leading-relaxed">
                A Diverse Ideation Engine. Think in branches, not in lines.
              </p>
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-2 md:justify-end items-center text-[13px]">
              <button onClick={onLaunchGauntlet} className="text-ink-1 hover:text-ink-0 transition-colors">
                Open test
              </button>
              <button onClick={onOpenWorkspace} className="text-ink-1 hover:text-ink-0 transition-colors">
                Workspace
              </button>
              <a href="#waitlist" className="text-ink-1 hover:text-ink-0 transition-colors">
                Waitlist
              </a>
              <a href="mailto:hello@conceptial.ai" className="text-ink-1 hover:text-ink-0 transition-colors">
                Contact
              </a>
            </div>
          </div>

          <MorphingWordmark className="block text-[clamp(64px,16vw,220px)] leading-[0.82] tracking-[-0.04em] text-ink-0" />

          <div className="mt-8 pt-6 border-t border-ink-line flex flex-col sm:flex-row items-baseline justify-between gap-3 text-ink-2 text-[11.5px]">
            <span>© 2026 Conceptial · Built on representation-engineered open-weight models.</span>
            <span className="font-mono">v0.4 · pollen×prism</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
