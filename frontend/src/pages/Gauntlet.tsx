import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Play, Zap } from 'lucide-react';
import { StreamWindow } from '../components/StreamWindow';
import { ScrollTree } from '../components/ScrollTree';
import mockData from '../mocks/mock_results.json';

const comparisons = mockData.comparisons;

const DEMO_PROJECTS = [
  { id: 'cmp-001', title: 'Legal-Tech GTM Strategy', tag: 'Strategy', prompt: comparisons[0].prompt },
  { id: 'cmp-002', title: 'Race Condition Debug', tag: 'Engineering', prompt: comparisons[1].prompt },
  { id: 'cmp-003', title: 'Zero-Trust Rollout', tag: 'Security', prompt: comparisons[2].prompt },
];

function getResultMetrics(idx: number) {
  if (idx === 0) return [
    { label: 'Distinct Archetypes', value: '+250%' },
    { label: 'Semantic Entropy', value: '+171.1%' },
    { label: 'PRISM Pass Rate', value: '100%' },
    { label: 'Vendi Score', value: '4.02' },
  ];
  if (idx === 1) return [
    { label: 'Pass@5', value: '100%' },
    { label: 'Relative Lift', value: '+25%' },
    { label: 'TTFT', value: '−26.7%' },
    { label: 'Vendi Score', value: '4.02' },
  ];
  return [
    { label: 'Vendi Score', value: '4.02' },
    { label: 'PRISM Pass Rate', value: '100%' },
    { label: 'Entropy', value: '4.32 bits' },
    { label: 'Near-Duplicates', value: '−18.4%' },
  ];
}

interface GauntletProps {
  onBack: () => void;
}

export function Gauntlet({ onBack }: GauntletProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [treeProgress, setTreeProgress] = useState(0);
  const treeAnimRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const result = comparisons[selectedIdx];

  const handleRun = useCallback(() => {
    setIsStreaming(true);
    setHasRun(true);
    setTreeProgress(0);

    // Animate tree 0→1 over ~2.6s
    const start = Date.now();
    const duration = 2600;
    treeAnimRef.current = setInterval(() => {
      const t = Math.min(1, (Date.now() - start) / duration);
      setTreeProgress(t);
      if (t >= 1 && treeAnimRef.current) {
        clearInterval(treeAnimRef.current);
        treeAnimRef.current = null;
      }
    }, 30);

    setTimeout(() => setIsStreaming(false), 2800);
  }, []);

  useEffect(() => {
    return () => {
      if (treeAnimRef.current) clearInterval(treeAnimRef.current);
    };
  }, []);

  const handleSelectProject = (idx: number) => {
    setSelectedIdx(idx);
    setHasRun(false);
    setTreeProgress(0);
  };

  return (
    <div className="gdemo">
      <div className="gauntlet__bg" />

      <header className="gauntlet__header">
        <div className="gauntlet__header-left">
          <button className="btn btn-ghost" onClick={onBack}>
            <ArrowLeft size={15} /> Back
          </button>
          <div className="gauntlet__brand">
            <div className="gauntlet__logo"><Zap size={14} /></div>
            <span className="gauntlet__title">Corpius 1</span>
            <span className="badge badge-brand">DEMO</span>
          </div>
        </div>
      </header>

      <main className="gdemo__main">
        {/* Left: project list */}
        <aside className="gdemo__sidebar">
          <p className="gdemo__sidebar-label">Demo Projects</p>
          <div className="gdemo__project-list">
            {DEMO_PROJECTS.map((p, i) => (
              <button
                key={p.id}
                className={`gdemo__project-card ${i === selectedIdx ? 'gdemo__project-card--active' : ''}`}
                onClick={() => handleSelectProject(i)}
              >
                <span className={`gdemo__project-tag gdemo__project-tag--${p.tag.toLowerCase()}`}>{p.tag}</span>
                <span className="gdemo__project-title">{p.title}</span>
                <span className="gdemo__project-prompt">{p.prompt}</span>
              </button>
            ))}
          </div>

          <motion.button
            className="gdemo__run-btn"
            onClick={handleRun}
            whileTap={{ scale: 0.97 }}
            disabled={isStreaming}
          >
            <Play size={15} />
            {isStreaming ? 'Generating...' : 'Run Demo'}
          </motion.button>
        </aside>

        {/* Right: tree + output */}
        <div className="gdemo__output">
          <div className="gdemo__tree-wrap">
            <ScrollTree progress={treeProgress} className="gdemo__tree" />
            {!hasRun && (
              <div className="gdemo__tree-hint">Select a project and hit Run Demo</div>
            )}
          </div>

          <AnimatePresence>
            {hasRun && (
              <motion.div
                className="gdemo__stream-section"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
              >
                <StreamWindow
                  text={result.steered.text}
                  isStreaming={isStreaming}
                  label={`Corpius 1  ·  Harmonic L20+L28+L36  ·  α=1.2`}
                />

                {!isStreaming && (
                  <motion.div
                    className="gauntlet__result-metrics"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.2 }}
                  >
                    {getResultMetrics(selectedIdx).map(m => (
                      <div key={m.label} className="gauntlet__result-metric">
                        <span className="gauntlet__result-metric-label">{m.label}</span>
                        <span className="gauntlet__result-metric-value">{m.value}</span>
                      </div>
                    ))}
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
