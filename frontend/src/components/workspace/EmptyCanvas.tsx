/**
 * EmptyCanvas — rendered in place of React Flow when the project has no root.
 *
 * Avoids the bug where RF.fitView shrinks a single tiny seed node to 10% zoom.
 * Presents a centered, generous "Start your map" prompt with a real textarea
 * and submit affordance. Once submitted, the parent swaps to the RF canvas.
 */
import { useCallback, useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, ArrowRight } from 'lucide-react';

interface Props {
  defaultText?: string;
  onSubmit: (text: string) => void;
}

const SUGGESTIONS = [
  'How might we improve onboarding so new users reach an aha moment in 60s?',
  'Generate 6 distinct strategic angles for a legal-tech startup launch.',
  'What\'s the right way to break a creative-block on naming a new product?',
];

export function EmptyCanvas({ defaultText = '', onSubmit }: Props) {
  const [text, setText] = useState(defaultText);
  const textRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textRef.current?.focus();
  }, []);

  const submit = useCallback(() => {
    const t = text.trim();
    if (!t) return;
    onSubmit(t);
  }, [text, onSubmit]);

  const onKey = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  }, [submit]);

  return (
    <div className="absolute inset-0 flex items-center justify-center overflow-y-auto px-6 py-10">
      {/* dot grid */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none opacity-50"
        style={{
          backgroundImage: `radial-gradient(circle, rgba(22,19,24,0.08) 1px, transparent 1.4px)`,
          backgroundSize: '28px 28px',
        }}
      />
      {/* radial vignette focus */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at 50% 45%, transparent 0%, transparent 30%, rgba(244, 239, 230, 0.7) 80%)',
        }}
      />

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 w-full max-w-2xl text-center"
      >
        <div className="inline-flex items-center gap-2 text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-5">
          <span className="h-px w-6 bg-ink-3" />
          New brainstorm
          <span className="h-px w-6 bg-ink-3" />
        </div>

        <h1
          className="font-display text-ink-0 text-[clamp(28px,3.4vw,46px)] leading-[1.05] tracking-[-0.02em] mb-3"
          style={{ fontVariationSettings: '"wght" 380, "opsz" 96' }}
        >
          Start with a <span className="italic">prompt.</span>
        </h1>
        <p className="text-ink-1 text-[14.5px] leading-relaxed max-w-md mx-auto mb-8">
          Describe the question or problem you want to explore.
          Conceptial will branch it into 6 distinct directions.
        </p>

        {/* Editor */}
        <div className="relative rounded-2xl bg-paper border border-[var(--ink-line-bold)] shadow-hover p-4 text-left">
          <div className="flex items-center gap-1.5 mb-2.5">
            <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full bg-brand/10 border border-brand/20">
              <span className="h-1 w-1 rounded-full bg-brand" />
              <span className="text-[9px] uppercase tracking-[0.16em] text-brand font-medium">Prompt</span>
            </span>
            <span className="text-[10px] text-ink-3 font-mono">root · d0</span>
            <span className="ml-auto text-[10px] text-ink-3 font-mono tabular-nums">
              {text.length} chars
            </span>
          </div>
          <textarea
            ref={textRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder="What do you want to brainstorm?"
            rows={3}
            className="w-full bg-transparent outline-none resize-none text-ink-0 text-[16px] leading-[1.6] placeholder:text-ink-3 font-display"
            style={{ fontVariationSettings: '"wght" 460, "opsz" 32' }}
          />
          <div className="mt-3 pt-3 border-t border-[var(--ink-line-soft)] flex items-center justify-between">
            <span className="text-[11px] text-ink-2 inline-flex items-center gap-1">
              <Sparkles size={11} className="text-brand" />
              Branches into 6 archetypes
            </span>
            <button
              onClick={submit}
              disabled={!text.trim()}
              className="inline-flex items-center gap-1.5 bg-brand hover:bg-brand-deep disabled:opacity-40 disabled:hover:bg-brand text-white text-[13px] font-medium px-4 py-2 rounded-full shadow-brand transition-all"
            >
              Start brainstorming
              <kbd className="text-[10px] opacity-80 ml-0.5 inline-flex items-center gap-0.5">⌘↵</kbd>
              <ArrowRight size={13} />
            </button>
          </div>
        </div>

        {/* Suggestions */}
        <div className="mt-7">
          <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-3">
            Or try one of these
          </div>
          <div className="grid grid-cols-1 gap-2">
            {SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                onClick={() => setText(s)}
                className="group text-left px-3.5 py-2.5 rounded-lg bg-paper-1 border border-[var(--ink-line)] hover:border-[var(--ink-line-bold)] hover:bg-paper-2 transition-all text-[12.5px] text-ink-1 leading-relaxed flex items-start gap-2"
              >
                <span className="h-1.5 w-1.5 rounded-full mt-1.5 shrink-0" style={{ background: `oklch(0.62 0.20 ${[280, 25, 200][i]})` }} />
                <span className="flex-1 group-hover:text-ink-0 transition-colors">{s}</span>
                <ArrowRight size={11} className="text-ink-3 mt-1 opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
