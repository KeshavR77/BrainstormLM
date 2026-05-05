import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Play, Zap } from 'lucide-react';
import { StreamWindow } from '../components/StreamWindow';
import mockData from '../mocks/mock_results.json';

const comparisons = mockData.comparisons;

const DEMO_PROJECTS = [
  { id: 'cmp-001', title: 'Legal-Tech GTM Strategy', tag: 'Strategy', prompt: comparisons[0].prompt },
  { id: 'cmp-002', title: 'Race Condition Debug', tag: 'Engineering', prompt: comparisons[1].prompt },
  { id: 'cmp-003', title: 'Zero-Trust Rollout', tag: 'Security', prompt: comparisons[2].prompt },
];

const DEMO_IDEA_MAPS = [
  [
    {
      title: 'Visionary Alignment',
      description: 'Prove value against GC risk, not feature volume.',
      metric: '4.15 Vendi',
    },
    {
      title: 'Trusted Advisor Leverage',
      description: 'Use existing advisory trust as the distribution moat.',
      metric: '8.5 coherence',
    },
    {
      title: 'Innovation Hubs',
      description: 'Co-develop inside client labs for credible references.',
      metric: 'Joint proof',
    },
    {
      title: 'Strategic Law-Firm Partners',
      description: 'Turn top-tier firm relationships into a proof engine.',
      metric: 'Enterprise pull',
    },
    {
      title: 'Phased Integration',
      description: 'Meet DMS, governance, and procurement where they are.',
      metric: 'Lower friction',
    },
  ],
  [
    {
      title: 'Failure Boundary',
      description: 'Narrow the bug to shared job state, not infrastructure.',
      metric: 'Root cause',
    },
    {
      title: 'Shared State Read',
      description: 'Two workers observe the same pending flag.',
      metric: 'Race point',
    },
    {
      title: 'Per-Key Lock',
      description: 'Guard read-modify-write by job key.',
      metric: 'Minimal fix',
    },
    {
      title: 'Atomic Compare-Swap',
      description: 'Fallback if the store supports conditional writes.',
      metric: 'Safer state',
    },
    {
      title: 'Idempotent Publish',
      description: 'Add dedupe protection around side effects.',
      metric: 'Resilience',
    },
  ],
  [
    {
      title: 'High-Risk Segments',
      description: 'Start where blast radius and exposure are highest.',
      metric: 'Phase 1',
    },
    {
      title: 'IAM Foundation',
      description: 'Unify identity before expanding controls.',
      metric: 'Control plane',
    },
    {
      title: 'Contractor ZTNA',
      description: 'Handle SaaS and acquired workforces in parallel.',
      metric: 'Fast wedge',
    },
    {
      title: 'Acquisition Templates',
      description: 'Repeatable rollout path for subsidiaries.',
      metric: 'Scale path',
    },
    {
      title: 'Progressive Trust KPIs',
      description: 'Measure trust boundaries, not tool deployment count.',
      metric: 'Executive signal',
    },
  ],
];

function getResultMetrics(idx: number) {
  if (idx === 0) return [
    { label: 'Archetypes vs Baseline', value: '+250%' },
    { label: 'Entropy vs Baseline', value: '+171.1%' },
    { label: 'Rollup Entropy Lift', value: '+18.2%' },
    { label: 'PRISM vs Unsteered', value: '+2.2pp' },
  ];
  if (idx === 1) return [
    { label: 'Pass@5 (5 tries, ≥1 correct)', value: '100%' },
    { label: 'vs Baseline Pass@5', value: '80%→100%' },
    { label: 'TTFT vs Baseline', value: '−26.7%' },
    { label: 'Output Token Tax', value: '0%' },
  ];
  return [
    { label: 'Vendi vs Control', value: '+5.7%' },
    { label: 'Near-Dup vs Control', value: '−18.4%' },
    { label: 'PRISM vs Unsteered', value: '+2.2pp' },
    { label: 'Entropy Rollup Lift', value: '+18.2%' },
  ];
}

interface GauntletProps {
  onBack: () => void;
}

function DemoIdeaMap({ prompt, nodes }: { prompt: string; nodes: typeof DEMO_IDEA_MAPS[number] }) {
  return (
    <div className="gdemo__map">
      <div className="gdemo__map-seed">
        <span className="gdemo__map-seed-label">Seed prompt</span>
        <strong>{prompt}</strong>
      </div>
      <div className="gdemo__map-rail" aria-hidden="true" />
      <div className="gdemo__map-grid">
        {nodes.map((node, idx) => (
          <motion.article
            key={node.title}
            className={`gdemo__map-node ${idx === 0 ? 'gdemo__map-node--selected' : ''}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: idx * 0.05 }}
          >
            <span className="gdemo__map-node-index">{String(idx + 1).padStart(2, '0')}</span>
            <h3>{node.title}</h3>
            <p>{node.description}</p>
            <span className="gdemo__map-node-metric">{node.metric}</span>
          </motion.article>
        ))}
      </div>
    </div>
  );
}

export function Gauntlet({ onBack }: GauntletProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);

  const result = comparisons[selectedIdx];
  const ideaNodes = DEMO_IDEA_MAPS[selectedIdx] ?? DEMO_IDEA_MAPS[0];

  const handleRun = useCallback(() => {
    setIsStreaming(true);
    setTimeout(() => setIsStreaming(false), 2800);
  }, []);

  const handleSelectProject = (idx: number) => {
    setSelectedIdx(idx);
    setIsStreaming(false);
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
            <span className="gauntlet__title">Conceptial Test</span>
            <span className="badge badge-brand">TEST</span>
          </div>
        </div>
      </header>

      <main className="gdemo__main">
        {/* Left: project list */}
        <aside className="gdemo__sidebar">
          <p className="gdemo__sidebar-label">Curated Scenarios</p>
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
            {isStreaming ? 'Replaying...' : 'Replay Test'}
          </motion.button>
        </aside>

        {/* Right: tree + output */}
        <div className="gdemo__output">
          <section className="gdemo__brief">
            <span className="gdemo__brief-label">Prepared prompt</span>
            <p>{result.prompt}</p>
          </section>

          <div className="gdemo__tree-wrap">
            <DemoIdeaMap prompt={result.prompt} nodes={ideaNodes} />
          </div>

          <AnimatePresence>
            <motion.div
              key={result.id}
              className="gdemo__stream-section"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.4 }}
            >
              <StreamWindow
                text={result.steered.text}
                isStreaming={isStreaming}
                label="Conceptial output"
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
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
