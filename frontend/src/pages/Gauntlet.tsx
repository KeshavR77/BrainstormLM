/**
 * Gauntlet — "Open test" demo page.
 *
 * Replays a recorded comparison between a standard LLM and Conceptial
 * on a strategy prompt, showing the resulting idea tree + a streamed
 * output and three headline metrics. Tailwind + new token palette.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Play, Pause, RotateCcw, Sparkles, ArrowRight } from 'lucide-react';
import mockData from '../mocks/mock_results.json';
import { Logo, Wordmark } from '../components/Logo';
import { depthHue, fadeRise, staggerContainer } from '../styles/motion';

const comparisons = mockData.comparisons;

const DEMO_IDEA_MAPS = [
  [
    { title: 'Visionary alignment',     description: 'Prove value against GC risk, not feature volume.',                 metric: 'Risk-first pitch' },
    { title: 'Trusted advisor leverage',description: 'Use existing advisory trust as the distribution moat.',           metric: 'Trust moat'       },
    { title: 'Innovation hubs',         description: 'Co-develop inside client labs for credible references.',           metric: 'Joint proof'      },
    { title: 'Strategic law-firm partners', description: 'Turn top-tier firm relationships into a proof engine.',        metric: 'Enterprise pull'  },
    { title: 'Phased integration',      description: 'Meet DMS, governance, and procurement where they are.',           metric: 'Lower friction'   },
  ],
];

const METRICS = [
  {
    label: 'More distinct strategies',
    value: '14 vs 4',
    note: 'Baseline collapses to 4 archetypes; Conceptial returns 14',
    hue: depthHue(0),
  },
  {
    label: 'More correct paths',
    value: '+25%',
    note: 'Pass@5 — 5 / 5 valid answers vs 4 / 5 baseline',
    hue: depthHue(1),
  },
  {
    label: 'Quality gate pass',
    value: '100%',
    note: 'Coherence + relevance held on every judged sample',
    hue: -1 /* brand */,
  },
];

const STRATEGY_VARIETY_NOTE =
  'Same prompt, same model — only the steering vector changes. The baseline LLM lands on one popular angle and a few near-duplicates; Conceptial fans into 14 separate strategic modes that each pass the quality gate.';

interface GauntletProps { onBack: () => void }

export function Gauntlet({ onBack }: GauntletProps) {
  const [selectedIdx]  = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedChars, setStreamedChars] = useState(0);

  const result = comparisons[selectedIdx];
  const ideaNodes = DEMO_IDEA_MAPS[selectedIdx] ?? DEMO_IDEA_MAPS[0];
  const fullText = result.steered.text;

  /* Typewriter */
  const tRef = useRef<number | null>(null);
  const stopTimer = () => { if (tRef.current) { window.clearTimeout(tRef.current); tRef.current = null; } };

  const startStream = useCallback(() => {
    stopTimer();
    setIsStreaming(true);
    setStreamedChars(0);
    let i = 0;
    const tick = () => {
      i = Math.min(i + Math.ceil(Math.random() * 3 + 1), fullText.length);
      setStreamedChars(i);
      if (i < fullText.length) tRef.current = window.setTimeout(tick, 14 + Math.random() * 18);
      else setIsStreaming(false);
    };
    tick();
  }, [fullText]);

  const pause = () => { stopTimer(); setIsStreaming(false); };
  const reset = () => { stopTimer(); setIsStreaming(false); setStreamedChars(0); };

  useEffect(() => () => stopTimer(), []);

  /* Auto-start once on mount */
  useEffect(() => {
    const t = window.setTimeout(startStream, 500);
    return () => window.clearTimeout(t);
  }, [startStream]);

  return (
    <div className="min-h-screen bg-paper text-ink-0 font-sans antialiased">
      {/* Nav */}
      <nav className="sticky top-0 z-30 backdrop-blur-md bg-paper/85 border-b border-[var(--ink-line)]">
        <div className="max-w-7xl mx-auto px-6 sm:px-10 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="inline-flex items-center gap-1.5 text-ink-1 hover:text-ink-0 text-sm transition-colors"
            >
              <ArrowLeft size={14} /> Home
            </button>
            <div className="h-4 w-px bg-[var(--ink-line)]" />
            <Wordmark size={19} weight={560} />
            <span className="hidden sm:inline-flex items-center gap-1.5 text-[10.5px] uppercase tracking-[0.18em] text-brand bg-brand/10 px-2 py-0.5 rounded-full">
              <span className="h-1 w-1 rounded-full bg-brand" />
              Demo
            </span>
          </div>
          <a
            href="#/app"
            className="bg-ink-0 hover:bg-ink-1 text-paper text-sm px-4 py-1.5 rounded-full font-medium transition-colors inline-flex items-center gap-1.5"
          >
            Open workspace <ArrowRight size={13} />
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-6 sm:px-10 pt-14 pb-6 max-w-7xl mx-auto">
        <div className="text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-3">A priori benchmark · replay</div>
        <h1
          className="font-display text-ink-0 text-[clamp(32px,4.4vw,56px)] leading-[1.04] tracking-[-0.02em] max-w-3xl"
          style={{ fontVariationSettings: '"wght" 380, "opsz" 96' }}
        >
          Same prompt, <span className="italic">five</span> meaningfully different strategies.
        </h1>
        <p className="mt-4 text-ink-1 text-[15.5px] max-w-2xl leading-relaxed">
          Watch Conceptial replay a strategy prompt that asked for go-to-market directions for a legal-tech startup. The
          baseline LLM converged on one popular angle. Conceptial returned a tree of distinct archetypes.
        </p>
      </section>

      {/* Prompt + controls */}
      <section className="px-6 sm:px-10 max-w-7xl mx-auto">
        <div className="rounded-2xl bg-paper-1 border border-[var(--ink-line)] overflow-hidden shadow-rest">
          <div className="px-5 py-4 border-b border-[var(--ink-line)] flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 min-w-0">
              <Sparkles size={14} className="text-brand shrink-0" />
              <span className="text-[10.5px] uppercase tracking-[0.18em] text-ink-3 shrink-0">Prompt</span>
              <span className="text-[13.5px] text-ink-1 truncate">{result.prompt}</span>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={isStreaming ? pause : startStream}
                className="inline-flex items-center gap-1 bg-brand hover:bg-brand-deep text-white text-[12px] font-medium px-3 py-1.5 rounded-full transition-colors"
              >
                {isStreaming ? <Pause size={11} /> : <Play size={11} />}
                {isStreaming ? 'Pause' : streamedChars > 0 ? 'Resume' : 'Replay'}
              </button>
              <button
                onClick={reset}
                title="Reset"
                className="p-1.5 rounded-full hover:bg-paper-2 text-ink-2 hover:text-ink-0 transition-colors"
              >
                <RotateCcw size={12} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Two-column: idea map + streamed output */}
      <section className="px-6 sm:px-10 max-w-7xl mx-auto mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Idea map */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="lg:col-span-7 rounded-2xl bg-paper-1 border border-[var(--ink-line)] p-6 shadow-rest"
        >
          <div className="flex items-baseline justify-between mb-5">
            <h2 className="font-display text-ink-0 text-[18px] tracking-tight" style={{ fontVariationSettings: '"wght" 500, "opsz" 32' }}>
              Idea map
            </h2>
            <span className="text-[10.5px] uppercase tracking-[0.18em] text-ink-3">{ideaNodes.length} archetypes</span>
          </div>

          {/* Seed → branches */}
          <div className="space-y-2.5">
            {ideaNodes.map((n, idx) => (
              <motion.article
                key={n.title}
                variants={fadeRise}
                className={`group flex gap-3 items-start rounded-xl p-3.5 border transition-all ${
                  idx === 0
                    ? 'bg-paper border-brand shadow-brand'
                    : 'bg-paper border-[var(--ink-line)] hover:border-[var(--ink-line-bold)]'
                }`}
                style={{ '--gen-hue': depthHue(idx) } as React.CSSProperties}
              >
                <div className="shrink-0 flex flex-col items-center gap-1.5 pt-0.5">
                  <span
                    className="h-6 w-6 rounded-full flex items-center justify-center text-[10px] text-white font-mono"
                    style={{ background: idx === 0 ? 'var(--brand)' : 'var(--gen)' }}
                  >
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                  {idx < ideaNodes.length - 1 && (
                    <span className="w-px h-3 bg-[var(--ink-line)]" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3
                      className="font-display text-ink-0 text-[14.5px] tracking-tight"
                      style={{ fontVariationSettings: '"wght" 540, "opsz" 24' }}
                    >
                      {n.title}
                    </h3>
                    {idx === 0 && (
                      <span className="text-[9px] uppercase tracking-[0.16em] text-brand bg-brand/10 px-1.5 py-0.5 rounded-full">
                        Selected
                      </span>
                    )}
                  </div>
                  <p className="text-[12.5px] text-ink-1 leading-relaxed">{n.description}</p>
                </div>
                <span className="shrink-0 text-[10.5px] font-mono text-ink-2 tabular-nums whitespace-nowrap">
                  {n.metric}
                </span>
              </motion.article>
            ))}
          </div>
        </motion.div>

        {/* Streamed output */}
        <div className="lg:col-span-5 rounded-2xl bg-paper-1 border border-[var(--ink-line)] p-6 shadow-rest flex flex-col min-h-[460px]">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="font-display text-ink-0 text-[18px] tracking-tight" style={{ fontVariationSettings: '"wght" 500, "opsz" 32' }}>
              Conceptial output
            </h2>
            <span className="text-[10.5px] uppercase tracking-[0.18em] text-ink-3 tabular-nums">
              {isStreaming ? 'Streaming…' : streamedChars === 0 ? 'Idle' : 'Complete'}
            </span>
          </div>
          <div className="flex-1 bg-paper border border-[var(--ink-line)] rounded-xl p-4 overflow-y-auto font-mono text-[12.5px] leading-[1.7] text-ink-1 whitespace-pre-wrap">
            {fullText.slice(0, streamedChars)}
            {isStreaming && (
              <span className="inline-block w-[1ch] align-baseline" style={{ background: 'var(--brand)', height: '1em', opacity: 0.85, animation: 'cgblink 1s steps(2) infinite' }} />
            )}
            {streamedChars === 0 && !isStreaming && (
              <span className="text-ink-3 italic">Press Replay to stream the recorded Conceptial output.</span>
            )}
          </div>

          {/* Metrics */}
          <AnimatePresence>
            {!isStreaming && streamedChars >= fullText.length && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="mt-4"
              >
                <div className="grid grid-cols-3 gap-2">
                  {METRICS.map((m) => (
                    <div
                      key={m.label}
                      className="rounded-lg bg-paper border border-[var(--ink-line)] px-3 py-2.5"
                      style={{ '--gen-hue': m.hue >= 0 ? m.hue : 0 } as React.CSSProperties}
                    >
                      <div className="flex items-center gap-1.5 mb-1">
                        <span
                          className="h-1 w-1 rounded-full"
                          style={{ background: m.hue < 0 ? 'var(--brand)' : 'var(--gen)' }}
                        />
                        <span className="text-[9px] uppercase tracking-[0.16em] text-ink-3">{m.label}</span>
                      </div>
                      <div
                        className="font-display tabular-nums text-[22px] leading-none tracking-tight text-ink-0"
                        style={{ fontVariationSettings: '"wght" 520, "opsz" 48' }}
                      >
                        {m.value}
                      </div>
                      <div className="mt-1.5 text-[10px] text-ink-3 leading-snug">{m.note}</div>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[11px] text-ink-2 leading-relaxed">{STRATEGY_VARIETY_NOTE}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {/* CTA strip */}
      <section className="px-6 sm:px-10 max-w-7xl mx-auto mt-12 mb-20">
        <div className="rounded-2xl bg-paper-2 border border-[var(--ink-line)] p-8 sm:p-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">
          <div className="max-w-xl">
            <div className="text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-2">Try it on your own prompt</div>
            <h3
              className="font-display text-ink-0 text-[clamp(22px,3vw,30px)] tracking-tight leading-tight"
              style={{ fontVariationSettings: '"wght" 480, "opsz" 48' }}
            >
              See what your idea looks like in branches.
            </h3>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <a
              href="#/app"
              className="inline-flex items-center gap-1.5 bg-brand hover:bg-brand-deep text-white text-[14px] font-medium px-5 py-3 rounded-full shadow-brand transition-all hover:translate-y-[-1px]"
            >
              Open workspace <ArrowRight size={14} />
            </a>
            <a
              href="#/"
              className="inline-flex items-center gap-1.5 text-ink-0 text-[14px] font-medium px-5 py-3 rounded-full border border-[var(--ink-line-bold)] hover:bg-paper transition-colors"
            >
              <Logo size={16} /> Back to home
            </a>
          </div>
        </div>
      </section>

      <style>{`@keyframes cgblink { 50% { opacity: 0; } }`}</style>
    </div>
  );
}
