/**
 * Demo.tsx — Conceptial vs Gemini-direct (the headline comparison).
 *
 * The cleanest experimental isolation we ship: same image renderer
 * (Gemini), same brief, only the upstream text-prompt source changes.
 * Left column: Conceptial 14B writes the descriptions, Gemini renders.
 * Right column: Gemini gets the user's brief directly, no upstream LLM.
 *
 * What you see: 8 distinct logos on the left vs 8 near-identical Stanford
 * seal/gear variants on the right. That gap IS Conceptial's contribution.
 *
 * GPT-4o and Claude Opus 4.7 comparisons moved to /frontier.
 */

import { ArrowLeft, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { Wordmark } from '../components/Logo';
import { fadeRise, staggerContainer } from '../styles/motion';

interface DemoProps {
  onBack: () => void;
  onOpenWorkspace: () => void;
}

interface DemoCard {
  src: string;
  alt: string;
}

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

const CONTROL_CARDS: DemoCard[] = Array.from({ length: 8 }, (_, i) => ({
  src: `/demo/control/${String(i + 1).padStart(2, '0')}.webp`,
  alt: `Gemini-direct rendering ${i + 1}: Stanford engineering club logo`,
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
          <span className={`inline-block w-2.5 h-2.5 rounded-full ${swatchClass}`} aria-hidden="true" />
          <h2
            className="font-display text-ink-0 text-[22px] tracking-tight"
            style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}
          >
            {title}
          </h2>
        </div>
        <span className="block mt-1.5 text-ink-3 text-[10px] tabular-nums uppercase tracking-[0.18em]">
          {subtitle}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-5">
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
            <figcaption className="mt-2 text-[11px] text-ink-2 tabular-nums">
              {armLabel} · #{i + 1}
            </figcaption>
          </motion.figure>
        ))}
      </div>
    </div>
  );
}

export function Demo({ onBack, onOpenWorkspace }: DemoProps) {
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
            <span className="text-ink-2 text-[12px] hidden sm:inline">· Demo</span>
          </div>

          <div className="flex items-center gap-1 sm:gap-2">
            <a
              href="#/frontier"
              className="text-ink-1 hover:text-ink-0 text-[13px] px-3 py-1.5 rounded-md transition-colors hidden sm:inline-block"
            >
              Frontier
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
          The headline comparison
        </div>
        <h1
          className="font-display text-ink-0 text-[clamp(34px,5vw,64px)] leading-[1.02] tracking-[-0.02em] mb-4"
          style={{ fontVariationSettings: '"wght" 380, "opsz" 144' }}
        >
          Same brief. Same renderer. <em className="not-italic text-brand">Different ideation engine.</em>
        </h1>
        <p className="text-ink-1 text-[15px] leading-relaxed max-w-2xl">
          The Gemini image model gets the same Stanford-engineering-club brief
          both ways. On the left, Conceptial 14B writes the visual descriptions
          first. On the right, Gemini sees the user's query directly — no
          upstream LLM. The visible gap is what Conceptial adds.
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
              · logo-stanford-engineering · N=8 per arm · same Gemini image model
            </span>
          </div>
          <p
            className="font-display text-ink-0 text-[18px] sm:text-[20px] leading-relaxed italic"
            style={{ fontVariationSettings: '"wght" 380, "opsz" 36' }}
          >
            “Give me a creative logo for an engineering club at Stanford.”
          </p>
        </aside>
      </section>

      {/* ── Comparison grid (2 columns, 8 cards each in 2×4 sub-grids) ── */}
      <main className="px-6 sm:px-10 pt-10 pb-32 max-w-7xl mx-auto">
        <motion.section
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12"
        >
          <Column
            title="Conceptial 14B → Gemini"
            subtitle="Upstream ideation + render"
            swatchClass="bg-brand"
            cards={CONCEPTIAL_CARDS}
            armLabel="Conceptial"
          />
          <Column
            title="Gemini direct"
            subtitle="Renderer alone — no upstream LLM"
            swatchClass="bg-ink-3"
            cards={CONTROL_CARDS}
            armLabel="Gemini"
          />
        </motion.section>

        {/* Footnote */}
        <p className="mt-12 text-ink-3 text-[12px] leading-relaxed max-w-2xl">
          Both sides use the same Gemini image model. The only thing that
          changes between columns is whether an upstream LLM (Conceptial 14B)
          writes a richer image-generation prompt first. For the cross-vendor
          comparison vs GPT-4o and Claude Opus 4.7, see{' '}
          <a href="#/frontier" className="text-ink-1 underline decoration-ink-3 underline-offset-2 hover:text-ink-0">the Frontier page</a>.
        </p>
      </main>
    </div>
  );
}
