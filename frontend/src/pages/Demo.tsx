/**
 * Demo.tsx — Conceptial vs Gemini-direct (the headline comparison).
 *
 * The cleanest experimental isolation we ship: same image renderer
 * (Gemini), same brief, only the upstream text-prompt source changes.
 * Conceptial 14B writes the descriptions, Gemini renders; Gemini-direct
 * gets the user's brief with no upstream LLM.
 *
 * Rather than two side-by-side columns, the two arms are shown one at a
 * time. A segmented toggle (with a sliding pill) flips between them, and
 * the panel slides / can be swiped left↔right. What you see: 8 distinct
 * logos for Conceptial vs 8 near-identical Stanford seal/gear variants for
 * Gemini-direct. That gap IS Conceptial's contribution.
 *
 * GPT-4o and Claude Opus 4.7 comparisons moved to /frontier.
 */

import { useCallback, useState } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { AnimatePresence, motion, type PanInfo } from 'framer-motion';
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

// The 8 (of 16) Gemini-direct renders that converge hardest: circular
// gear-ring "seal" variants on light backgrounds — same Hoover-tower / block-S
// / gear motif every time. The visual sameness IS the point of this arm.
const CONTROL_FILENAMES = ['04', '07', '08', '09', '10', '14', '15', '16'];

const CONCEPTIAL_CARDS: DemoCard[] = CONCEPT_NAMES.map((name, i) => ({
  src: `/demo/conceptial/${CONCEPTIAL_FILENAMES[i]}.webp`,
  alt: `Conceptial 14B direction ${i + 1}, rendered by Gemini: ${name}`,
}));

const CONTROL_CARDS: DemoCard[] = CONTROL_FILENAMES.map((file, i) => ({
  src: `/demo/control/${file}.webp`,
  alt: `Gemini-direct rendering ${i + 1}: Stanford engineering club logo`,
}));

interface Arm {
  key: string;
  label: string;
  title: string;
  subtitle: string;
  swatchClass: string;
  cards: DemoCard[];
  armLabel: string;
}

const ARMS: Arm[] = [
  {
    key: 'conceptial',
    label: 'Conceptial',
    title: 'Conceptial 14B → Gemini',
    subtitle: 'Upstream ideation + render',
    swatchClass: 'bg-brand',
    cards: CONCEPTIAL_CARDS,
    armLabel: 'Conceptial',
  },
  {
    key: 'gemini',
    label: 'Gemini direct',
    title: 'Gemini direct',
    subtitle: 'Renderer alone — no upstream LLM',
    swatchClass: 'bg-ink-3',
    cards: CONTROL_CARDS,
    armLabel: 'Gemini',
  },
];

const SWIPE_THRESHOLD = 60;

const panelVariants = {
  enter: (dir: number) => ({ x: dir >= 0 ? 48 : -48, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir: number) => ({ x: dir >= 0 ? -48 : 48, opacity: 0 }),
};

function ArmPanel({ arm }: { arm: Arm }) {
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2.5">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${arm.swatchClass}`} aria-hidden="true" />
        <h2
          className="font-display text-ink-0 text-[22px] tracking-tight"
          style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}
        >
          {arm.title}
        </h2>
        <span className="text-ink-3 text-[10px] tabular-nums uppercase tracking-[0.18em]">
          {arm.subtitle}
        </span>
      </div>

      <motion.div
        key={arm.key}
        variants={staggerContainer}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-5"
      >
        {arm.cards.map((card, i) => (
          <motion.figure key={card.src} variants={fadeRise} className="group">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-[var(--ink-line)] bg-paper-1">
              <img
                src={card.src}
                alt={card.alt}
                width={1024}
                height={1024}
                loading="eager"
                decoding="async"
                draggable={false}
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-[1.015] select-none"
              />
            </div>
            <figcaption className="mt-2 text-[11px] text-ink-2 tabular-nums">
              {arm.armLabel} · #{i + 1}
            </figcaption>
          </motion.figure>
        ))}
      </motion.div>
    </div>
  );
}

export function Demo({ onBack, onOpenWorkspace }: DemoProps) {
  const [active, setActive] = useState(0);
  const [dir, setDir] = useState(0);

  const select = useCallback(
    (next: number) => {
      setDir(next - active);
      setActive(next);
    },
    [active],
  );

  const handleDragEnd = useCallback(
    (_e: unknown, info: PanInfo) => {
      if (info.offset.x <= -SWIPE_THRESHOLD && active < ARMS.length - 1) {
        select(active + 1);
      } else if (info.offset.x >= SWIPE_THRESHOLD && active > 0) {
        select(active - 1);
      }
    },
    [active, select],
  );

  const arm = ARMS[active];

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
          both ways. With Conceptial 14B, an upstream LLM writes the visual
          descriptions first; with Gemini direct, the model sees the user's
          query alone — no upstream LLM. Flip between the two and the gap is
          what Conceptial adds.
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

      {/* ── Single-arm view with toggle + slide/swipe ───────── */}
      <main className="px-6 sm:px-10 pt-10 pb-32 max-w-7xl mx-auto">
        {/* Segmented toggle */}
        <div className="flex flex-col items-center gap-3 mb-8">
          <div
            role="tablist"
            aria-label="Choose ideation engine"
            className="relative inline-flex items-center rounded-full border border-[var(--ink-line)] bg-paper-1 p-1"
          >
            {ARMS.map((a, i) => {
              const isActive = i === active;
              return (
                <button
                  key={a.key}
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => select(i)}
                  className={`relative px-4 sm:px-5 py-2 rounded-full text-[13px] font-medium transition-colors ${
                    isActive ? 'text-paper' : 'text-ink-1 hover:text-ink-0'
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="arm-pill"
                      className="absolute inset-0 rounded-full bg-ink-0"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className="relative z-10 inline-flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${a.swatchClass}`} aria-hidden="true" />
                    {a.label}
                  </span>
                </button>
              );
            })}
          </div>
          <span className="text-ink-3 text-[11px]">Tap a label or swipe to compare</span>
        </div>

        {/* Sliding / swipeable panel */}
        <div className="overflow-hidden">
          <AnimatePresence mode="wait" custom={dir} initial={false}>
            <motion.div
              key={arm.key}
              custom={dir}
              variants={panelVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
              drag="x"
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={0.18}
              onDragEnd={handleDragEnd}
              className="cursor-grab active:cursor-grabbing"
            >
              <ArmPanel arm={arm} />
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Dot indicators */}
        <div className="mt-8 flex justify-center gap-2">
          {ARMS.map((a, i) => (
            <button
              key={a.key}
              aria-label={`Show ${a.label}`}
              onClick={() => select(i)}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === active ? 'w-6 bg-ink-0' : 'w-1.5 bg-ink-3 hover:bg-ink-2'
              }`}
            />
          ))}
        </div>

        {/* Footnote */}
        <p className="mt-12 text-ink-3 text-[12px] leading-relaxed max-w-2xl mx-auto text-center">
          Both arms use the same Gemini image model. The only thing that
          changes is whether an upstream LLM (Conceptial 14B) writes a richer
          image-generation prompt first. For the cross-vendor comparison vs
          GPT-4o and Claude Opus 4.7, see{' '}
          <a href="#/frontier" className="text-ink-1 underline decoration-ink-3 underline-offset-2 hover:text-ink-0">the Frontier page</a>.
        </p>
      </main>
    </div>
  );
}
