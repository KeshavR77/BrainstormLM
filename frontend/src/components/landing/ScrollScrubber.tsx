/**
 * ScrollScrubber — scroll-driven before/after split between "linear LLM"
 * (flat list of similar ideas) and "Conceptial" (branching tree). The split
 * line sweeps right as the user scrolls past this section.
 */
import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';

const LINEAR_IDEAS = [
  'Add a personalized welcome flow.',
  'Add a guided welcome flow with examples.',
  'Add a welcome flow that highlights features.',
  'Add a personalized walkthrough with sample data.',
  'Add a guided personalized welcome experience.',
];

const TREE_BRANCHES = [
  { tag: 'Personal payoff',    detail: 'Start from the user\'s goal, not the product tour.' },
  { tag: 'Progressive trust',  detail: 'Prove value before asking for setup work.' },
  { tag: 'Social proof first', detail: 'Show what similar teams achieved first.' },
  { tag: 'Role-based path',    detail: 'Designer, founder, or PM each gets a different first win.' },
];

export function ScrollScrubber() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start 80%', 'end 20%'],
  });
  const splitX = useTransform(scrollYProgress, [0, 1], ['10%', '85%']);
  const splitLineX = useTransform(scrollYProgress, [0, 1], ['10%', '85%']);

  return (
    <div ref={ref} className="relative w-full max-w-5xl mx-auto">
      <div className="grid grid-cols-2 gap-0 rounded-xl overflow-hidden border border-ink-line bg-paper-1">
        {/* Linear side */}
        <div className="relative p-8 sm:p-10 bg-paper-2">
          <div className="text-ink-2 text-[11px] uppercase tracking-[0.18em] mb-4">Linear LLM</div>
          <ul className="space-y-3">
            {LINEAR_IDEAS.map((idea, i) => (
              <li key={i} className="flex items-start gap-3 text-ink-1 text-sm leading-relaxed">
                <span className="mt-2 h-1 w-1 rounded-full bg-ink-3 flex-shrink-0" />
                {idea}
              </li>
            ))}
          </ul>
          <div className="mt-6 inline-flex items-center gap-2 text-[11px] text-ink-2">
            <span className="h-1.5 w-1.5 rounded-full bg-ink-3" />
            8 of 10 distinct · 7% near-duplicate rate
          </div>
        </div>

        {/* Conceptial side */}
        <div className="relative p-8 sm:p-10 bg-paper">
          <div className="text-ink-2 text-[11px] uppercase tracking-[0.18em] mb-4 flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-brand" />
            Conceptial
          </div>
          <ul className="space-y-3">
            {TREE_BRANCHES.map((b, i) => (
              <li
                key={i}
                className="rounded-md p-3 border border-ink-line bg-paper"
                style={{ '--gen-hue': [280, 320, 25, 140][i % 4] } as React.CSSProperties}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--gen)' }} />
                  <span className="text-ink-0 text-[12px] font-medium tracking-tight">{b.tag}</span>
                </div>
                <div className="text-ink-1 text-[12.5px] leading-snug pl-3.5">{b.detail}</div>
              </li>
            ))}
          </ul>
          <div className="mt-6 inline-flex items-center gap-2 text-[11px] text-ink-2">
            <span className="h-1.5 w-1.5 rounded-full bg-brand" />
            10 of 10 distinct · 1% near-duplicate rate
          </div>
        </div>
      </div>

      {/* Sweeping split line */}
      <motion.div
        aria-hidden
        className="absolute top-0 bottom-0 w-px pointer-events-none"
        style={{ left: splitLineX, background: 'var(--brand)' }}
      />
      <motion.div
        aria-hidden
        className="absolute -top-2 h-3 w-3 rounded-full pointer-events-none"
        style={{ left: splitX, background: 'var(--brand)', transform: 'translateX(-50%)', boxShadow: 'var(--shadow-brand)' }}
      />
    </div>
  );
}
