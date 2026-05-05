import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Copy, Check } from 'lucide-react';
import mockData from '../mocks/mock_results.json';

const comparisons = mockData.comparisons;

// Pull real metric labels per comparison from actual measured data
function getMetrics(cmp: typeof comparisons[number]) {
  const m = cmp.metrics as unknown as Record<string, number>;
  const chips: { label: string; value: string }[] = [];

  if (m.archetype_lift_pct != null)
    chips.push({ label: 'Strategy variety', value: `+${Math.round(m.archetype_lift_pct)}%` });
  if (m.entropy_lift_pct != null)
    chips.push({ label: 'Idea spread', value: `+${Math.round(m.entropy_lift_pct)}%` });
  if (m.pass_at_5_lift_pct != null)
    chips.push({ label: 'Correct paths', value: `+${Math.round(m.pass_at_5_lift_pct)}%` });
  if (m.vendi_score != null)
    chips.push({ label: 'Diversity score', value: m.vendi_score.toFixed(2) });
  if (m.prism_pass_rate != null)
    chips.push({ label: 'Quality gate', value: `${(m.prism_pass_rate * 100).toFixed(0)}%` });
  if (m.near_dup_corpius != null && m.near_dup_baseline != null) {
    const drop = Math.round((m.near_dup_baseline - m.near_dup_corpius) / m.near_dup_baseline * 100);
    chips.push({ label: 'Repeat answers', value: `−${drop}%` });
  }

  return chips;
}

export function ComparisonViewer() {
  const [idx, setIdx] = useState(0);
  const [copied, setCopied] = useState<'left' | 'right' | null>(null);
  const cmp = comparisons[idx];
  const chips = getMetrics(cmp);

  const copyText = (text: string, side: 'left' | 'right') => {
    navigator.clipboard.writeText(text);
    setCopied(side);
    setTimeout(() => setCopied(null), 1500);
  };

  return (
    <div className="cv">
      <div className="cv__prompt">
        <span className="cv__prompt-label">Prompt</span>
        <p className="cv__prompt-text">{cmp.prompt}</p>
      </div>

      <div className="cv__grid">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${cmp.id}-std`}
            className="cv__card cv__card--standard"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="cv__card-header">
              <span className="cv__card-badge cv__card-badge--muted">Standard LLM</span>
              <button className="cv__copy" onClick={() => copyText(cmp.standard.text, 'left')} aria-label="Copy">
                {copied === 'left' ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            <p className="cv__card-text">{cmp.standard.text}</p>
          </motion.div>
        </AnimatePresence>

        <div className="cv__divider"><span className="cv__vs">VS</span></div>

        <AnimatePresence mode="wait">
          <motion.div
            key={`${cmp.id}-steered`}
            className="cv__card cv__card--steered"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="cv__card-header">
              <span className="cv__card-badge cv__card-badge--gold">Corpius 1</span>
              <button className="cv__copy" onClick={() => copyText(cmp.steered.text, 'right')} aria-label="Copy">
                {copied === 'right' ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            <p className="cv__card-text">{cmp.steered.text}</p>
            {chips.length > 0 && (
              <div className="cv__metrics-row">
                {chips.map(c => (
                  <span key={c.label} className="cv__metric">
                    {c.label} <strong>{c.value}</strong>
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="cv__nav">
        <button className="cv__nav-btn" onClick={() => setIdx(i => (i - 1 + comparisons.length) % comparisons.length)} aria-label="Previous">
          <ChevronLeft size={16} />
        </button>
        <div className="cv__dots">
          {comparisons.map((_, i) => (
            <button key={i} className={`cv__dot ${i === idx ? 'cv__dot--active' : ''}`} onClick={() => setIdx(i)} aria-label={`Example ${i + 1}`} />
          ))}
        </div>
        <button className="cv__nav-btn" onClick={() => setIdx(i => (i + 1) % comparisons.length)} aria-label="Next">
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
