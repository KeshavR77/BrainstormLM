import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Copy, Check } from 'lucide-react';
import mockData from '../mocks/mock_results.json';

const comparisons = mockData.comparisons;

export function ComparisonViewer() {
  const [idx, setIdx] = useState(0);
  const [copied, setCopied] = useState<'left' | 'right' | null>(null);
  const cmp = comparisons[idx];

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
              <button
                className="cv__copy"
                onClick={() => copyText(cmp.standard.text, 'left')}
                aria-label="Copy"
              >
                {copied === 'left' ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            <p className="cv__card-text">{cmp.standard.text}</p>
          </motion.div>
        </AnimatePresence>

        <div className="cv__divider">
          <span className="cv__vs">VS</span>
        </div>

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
              <span className="cv__card-badge cv__card-badge--gold">Steered (Corpus 1)</span>
              <button
                className="cv__copy"
                onClick={() => copyText(cmp.steered.text, 'right')}
                aria-label="Copy"
              >
                {copied === 'right' ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            <p className="cv__card-text">{cmp.steered.text}</p>
            <div className="cv__metrics-row">
              <span className="cv__metric">
                Vendi <strong>{cmp.metrics.vendi_score.toFixed(2)}</strong>
              </span>
              <span className="cv__metric">
                PRISM <strong>{(cmp.metrics.prism_coherence * 100).toFixed(0)}%</strong>
              </span>
              <span className="cv__metric">
                Novelty <strong>{(cmp.metrics.novelty * 100).toFixed(0)}%</strong>
              </span>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="cv__nav">
        <button
          className="cv__nav-btn"
          onClick={() => setIdx(i => (i - 1 + comparisons.length) % comparisons.length)}
          aria-label="Previous"
        >
          <ChevronLeft size={16} />
        </button>
        <div className="cv__dots">
          {comparisons.map((_, i) => (
            <button
              key={i}
              className={`cv__dot ${i === idx ? 'cv__dot--active' : ''}`}
              onClick={() => setIdx(i)}
              aria-label={`Example ${i + 1}`}
            />
          ))}
        </div>
        <button
          className="cv__nav-btn"
          onClick={() => setIdx(i => (i + 1) % comparisons.length)}
          aria-label="Next"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}
