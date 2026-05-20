/**
 * Frontier.tsx — Cross-vendor logo comparison (formerly Demo.tsx).
 *
 * Three columns: GPT-4o, Claude Opus 4.7, Conceptial 14B. Same Stanford
 * engineering club logo brief. All renders go through the same Gemini
 * image model, so any visual difference traces to the upstream text-prompt
 * generator, not the renderer.
 */

import { ArrowLeft, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { Wordmark } from '../components/Logo';
import { fadeRise, staggerContainer } from '../styles/motion';

interface FrontierProps {
  onBack: () => void;
  onOpenWorkspace: () => void;
}

interface DemoCard {
  src: string;
  alt: string;
}

const GPT_CARDS: DemoCard[] = Array.from({ length: 8 }, (_, i) => ({
  src: `/demo/gpt/${String(i + 1).padStart(2, '0')}.webp`,
  alt: `GPT-4o direction ${i + 1}, rendered by Gemini: Stanford engineering club logo`,
}));

const CLAUDE_CARDS: DemoCard[] = Array.from({ length: 8 }, (_, i) => ({
  src: `/demo/anthropic/${String(i + 1).padStart(2, '0')}.webp`,
  alt: `Claude Opus 4.7 direction ${i + 1}, rendered by Gemini: Stanford engineering club logo`,
}));

const CONCEPT_NAMES = [
  "Engineer's Nexus",
  '3D Blueprint Vision',
  '3D Circuit Bloom',
  'Quantum Gear Revolution',
  'Innovate Core',
  'Infinite Blueprint Maze',
  'Bridge to Innovation',
  'Inventive Horizon',
];

const CONCEPTIAL_FILENAMES = ['01', '02', '03b', '04', '05b', '06b', '07', '08'];

const CONCEPTIAL_CARDS: DemoCard[] = CONCEPT_NAMES.map((name, i) => ({
  src: `/demo/conceptial/${CONCEPTIAL_FILENAMES[i]}.webp`,
  alt: `Conceptial 14B direction ${i + 1}, rendered by Gemini: ${name}`,
}));

interface ColumnProps {
  title: string;
  subtitle: string;
  swatchClass: string;
  cards: DemoCard[];
  armLabel: string;
}

function Column({ title, subtitle, swatchClass, cards, armLabel }: ColumnProps) {
  return (
    <div className="space-y-4">
      <div className="border-b border-[var(--ink-line)] pb-3">
        <div className="flex items-center gap-2.5">
          <span className={`inline-block w-2 h-2 rounded-full ${swatchClass}`} aria-hidden="true" />
          <h2
            className="font-display text-ink-0 text-[19px] tracking-tight"
            style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}
          >
            {title}
          </h2>
        </div>
        <span className="block mt-1.5 text-ink-3 text-[10px] tabular-nums uppercase tracking-[0.18em]">
          {subtitle}
        </span>
      </div>

      <div className="space-y-5">
        {cards.map((card, i) => (
          <motion.figure key={card.src} variants={fadeRise} className="group">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-[var(--ink-line)] bg-paper-1">
              <img
                src={card.src}
                alt={card.alt}
                width={1024}
                height={1024}
                loading="eager"
                decoding="sync"
                fetchPriority="high"
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-[1.015]"
              />
            </div>
            <figcaption className="mt-2 text-[12px] text-ink-2 tabular-nums">
              {armLabel} · #{i + 1}
            </figcaption>
          </motion.figure>
        ))}
      </div>
    </div>
  );
}

export function Frontier({ onBack, onOpenWorkspace }: FrontierProps) {
  return (
    <div className="min-h-screen bg-paper text-ink-0 font-sans antialiased">
      {/* ── Top Bar ──────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 backdrop-blur-md bg-paper/80 border-b border-[var(--ink-line)]">
        <div className="max-w-7xl mx-auto px-6 sm:px-10 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="inline-flex items-center gap-1.5 text-ink-1 hover:text-ink-0 text-sm transition-colors"
            >
              <ArrowLeft size={14} /> Home
            </button>
            <div className="h-4 w-px bg-[var(--ink-line)]" />
            <Wordmark size={19} weight={560} />
            <span className="text-ink-2 text-[12px] hidden sm:inline">· Frontier</span>
          </div>

          <div className="flex items-center gap-1 sm:gap-2">
            <a
              href="#/demo"
              className="text-ink-1 hover:text-ink-0 text-[13px] px-3 py-1.5 rounded-md transition-colors hidden sm:inline-block"
            >
              Demo
            </a>
            <button
              onClick={onOpenWorkspace}
              className="bg-ink-0 text-paper text-[13px] font-medium px-4 py-2 rounded-full hover:bg-ink-1 transition-colors inline-flex items-center gap-1.5"
            >
              Workspace <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────── */}
      <section className="px-6 sm:px-10 pt-12 sm:pt-16 pb-8 max-w-7xl mx-auto">
        <div className="text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-3">
          Frontier · cross-vendor comparison
        </div>
        <h1
          className="font-display text-ink-0 text-[clamp(34px,5vw,64px)] leading-[1.02] tracking-[-0.02em] mb-4"
          style={{ fontVariationSettings: '"wght" 380, "opsz" 144' }}
        >
          Same brief. <em className="not-italic text-brand">Three answers.</em>
        </h1>
        <p className="text-ink-1 text-[15px] leading-relaxed max-w-2xl">
          Eight directions each from GPT-4o, Claude Opus 4.7, and Conceptial
          14B. Images rendered by the same Gemini image model.
        </p>
      </section>

      {/* ── Brief callout ───────────────────────────────────── */}
      <section className="px-6 sm:px-10 max-w-7xl mx-auto">
        <aside className="border border-[var(--ink-line)] bg-paper-1/60 rounded-2xl px-5 sm:px-6 py-5 text-ink-1">
          <div className="flex items-baseline gap-3 mb-2">
            <span className="text-ink-3 text-[10px] tracking-[0.22em] uppercase">
              The brief
            </span>
            <span className="text-ink-3 text-[11px]">
              · logo-stanford-engineering · N=8 per arm · rendered by Gemini
            </span>
          </div>
          <p
            className="font-display text-ink-0 text-[18px] sm:text-[20px] leading-relaxed italic"
            style={{ fontVariationSettings: '"wght" 380, "opsz" 36' }}
          >
            “Give me a creative logo for an engineering club at Stanford. I
            want a three-sentence visual description suitable for an
            image-generation prompt.”
          </p>
        </aside>
      </section>

      {/* ── Comparison grid ─────────────────────────────────── */}
      <main className="px-6 sm:px-10 pt-10 pb-32 max-w-7xl mx-auto">
        <motion.section
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-10"
        >
          <Column
            title="GPT-4o"
            subtitle="8 directions"
            swatchClass="bg-emerald-500/70"
            cards={GPT_CARDS}
            armLabel="GPT-4o"
          />
          <Column
            title="Claude Opus 4.7"
            subtitle="8 directions"
            swatchClass="bg-amber-500/80"
            cards={CLAUDE_CARDS}
            armLabel="Opus 4.7"
          />
          <Column
            title="Conceptial 14B"
            subtitle="8 directions"
            swatchClass="bg-brand"
            cards={CONCEPTIAL_CARDS}
            armLabel="Conceptial"
          />
        </motion.section>

        {/* Footnote */}
        <p className="mt-12 text-ink-3 text-[12px] leading-relaxed max-w-2xl">
          Independent single-shot generations of the same brief. For the
          renderer-only baseline (no upstream LLM) see{' '}
          <a href="#/control" className="text-ink-1 underline decoration-ink-3 underline-offset-2 hover:text-ink-0">Gemini alone</a>{' '}
          — or the cleaner two-way{' '}
          <a href="#/demo" className="text-ink-1 underline decoration-ink-3 underline-offset-2 hover:text-ink-0">Conceptial vs Gemini-direct</a> comparison.
        </p>
      </main>
    </div>
  );
}
