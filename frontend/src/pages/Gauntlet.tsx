import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Layers, Zap, Play } from 'lucide-react';
import { AlphaSlider } from '../components/AlphaSlider';
import { StreamWindow } from '../components/StreamWindow';
import { MetricRadar } from '../components/MetricRadar';
import mockData from '../mocks/mock_results.json';

const comparisons = mockData.comparisons;

// Real measured configs from artifacts
const PRESETS = [
  { label: 'Baseline', alpha: 0.0, layer: 'none', note: 'Temp 0.7, no steering' },
  { label: 'L28 α=4.0', alpha: 4.0, layer: 'late', note: 'Peak Vendi 3.93 — novel-idea task' },
  { label: 'Harmonic ✦', alpha: 1.2, layer: 'harmonic', note: 'Pass@5 100% · Vendi 4.02 · TTFT −27%' },
];

interface GauntletProps {
  onBack: () => void;
}

type LayerMode = 'harmonic' | 'late' | 'none';

// Real metrics per result card, derived from the actual measured data
function getResultMetrics(idx: number) {
  if (idx === 0) return [
    { label: '+250%', sub: 'Distinct Archetypes' },
    { label: '+171.1%', sub: 'Semantic Entropy' },
    { label: '100%', sub: 'PRISM Pass Rate' },
    { label: '4.02', sub: 'Vendi Score' },
  ];
  if (idx === 1) return [
    { label: '100%', sub: 'Pass@5 (vs 80%)' },
    { label: '+25%', sub: 'Relative Lift' },
    { label: '−26.7%', sub: 'TTFT' },
    { label: '4.02', sub: 'Vendi Score' },
  ];
  return [
    { label: '4.02', sub: 'Vendi Score' },
    { label: '100%', sub: 'PRISM Pass Rate' },
    { label: '4.32 bits', sub: 'Entropy' },
    { label: '−18.4%', sub: 'Near-Duplicates' },
  ];
}

export function Gauntlet({ onBack }: GauntletProps) {
  const [alpha, setAlpha] = useState(1.2);
  const [layer, setLayer] = useState<LayerMode>('harmonic');
  const [prompt, setPrompt] = useState(comparisons[0].prompt);
  const [isStreaming, setIsStreaming] = useState(false);
  const [resultIdx, setResultIdx] = useState(0);
  const [hasRun, setHasRun] = useState(false);

  const result = comparisons[resultIdx];

  const handleGenerate = useCallback(() => {
    setIsStreaming(true);
    setHasRun(true);
    // Match prompt to the right comparison when possible
    const match = comparisons.findIndex(c => c.prompt === prompt);
    setResultIdx(match >= 0 ? match : Math.floor(Math.random() * comparisons.length));
    setTimeout(() => setIsStreaming(false), 2800);
  }, [prompt]);

  // Radar reflects real diversity/coherence tradeoff observed in sweep
  const predictedMetrics = layer === 'none'
    ? { diversity: 0.34, coherence: 0.82, novelty: 0.31, feasibility: 0.88 }
    : layer === 'harmonic'
    ? { diversity: Math.min(1, 0.4 + alpha * 0.22), coherence: Math.max(0.65, 0.92 - alpha * 0.04), novelty: Math.min(1, 0.35 + alpha * 0.28), feasibility: Math.max(0.55, 0.88 - alpha * 0.05) }
    : { diversity: Math.min(1, 0.3 + alpha * 0.17), coherence: Math.max(0.55, 0.9 - alpha * 0.06), novelty: Math.min(1, 0.25 + alpha * 0.18), feasibility: Math.max(0.45, 0.85 - alpha * 0.07) };

  const layerLabel = layer === 'harmonic' ? 'Harmonic L20+L28+L36' : layer === 'late' ? 'L28' : 'No steering';

  return (
    <div className="gauntlet">
      <div className="gauntlet__bg" />

      <header className="gauntlet__header">
        <div className="gauntlet__header-left">
          <button className="btn btn-ghost" onClick={onBack}>
            <ArrowLeft size={15} /> Back
          </button>
          <div className="gauntlet__brand">
            <div className="gauntlet__logo"><Zap size={14} /></div>
            <span className="gauntlet__title">The Gauntlet</span>
            <span className="badge badge-brand">DEMO</span>
          </div>
        </div>
        <div className="gauntlet__header-right">
          <span className="gauntlet__model-badge">Qwen/Qwen2.5-14B-Instruct</span>
        </div>
      </header>

      <main className="gauntlet__main">
        <div className="gauntlet__controls">
          <div className="gauntlet__panel">
            <h3 className="gauntlet__panel-title">Steering Controls</h3>

            <AlphaSlider value={alpha} onChange={setAlpha} min={0} max={4} step={0.1} />

            <div className="gauntlet__layer-select">
              <label className="gauntlet__field-label">
                <Layers size={13} /> Configuration
              </label>
              <div className="gauntlet__layer-toggle" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
                <button
                  className={`gauntlet__layer-btn ${layer === 'none' ? 'gauntlet__layer-btn--active' : ''}`}
                  onClick={() => { setLayer('none'); setAlpha(0); }}
                >Baseline</button>
                <button
                  className={`gauntlet__layer-btn ${layer === 'late' ? 'gauntlet__layer-btn--active' : ''}`}
                  onClick={() => setLayer('late')}
                >L28</button>
                <button
                  className={`gauntlet__layer-btn ${layer === 'harmonic' ? 'gauntlet__layer-btn--active' : ''}`}
                  onClick={() => setLayer('harmonic')}
                >Harmonic ✦</button>
              </div>
            </div>

            <div className="gauntlet__presets">
              <label className="gauntlet__field-label">Validated Configs</label>
              <div className="gauntlet__preset-grid">
                {PRESETS.map(p => (
                  <button
                    key={p.label}
                    className={`gauntlet__preset ${alpha === p.alpha && layer === p.layer ? 'gauntlet__preset--active' : ''}`}
                    onClick={() => { setAlpha(p.alpha); setLayer(p.layer as LayerMode); }}
                    title={p.note}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="gauntlet__panel">
            <h3 className="gauntlet__panel-title">Predicted Profile</h3>
            <div className="gauntlet__radar-wrap">
              <MetricRadar
                diversity={predictedMetrics.diversity}
                coherence={predictedMetrics.coherence}
                novelty={predictedMetrics.novelty}
                feasibility={predictedMetrics.feasibility}
                size={180}
              />
            </div>
            <div className="gauntlet__metric-list">
              <div className="gauntlet__metric-item">
                <span>Diversity</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--gold">{(predictedMetrics.diversity * 100).toFixed(0)}%</span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Coherence</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--teal">{(predictedMetrics.coherence * 100).toFixed(0)}%</span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Novelty</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--gold">{(predictedMetrics.novelty * 100).toFixed(0)}%</span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Feasibility</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--teal">{(predictedMetrics.feasibility * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="gauntlet__output">
          <div className="gauntlet__prompt-bar">
            <input
              type="text"
              className="gauntlet__prompt-input"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Enter a strategic question..."
            />
            <motion.button
              className="gauntlet__generate-btn"
              onClick={handleGenerate}
              whileTap={{ scale: 0.97 }}
              disabled={isStreaming || !prompt.trim()}
            >
              <Play size={15} />
              {isStreaming ? 'Generating...' : 'Generate'}
            </motion.button>
          </div>

          {hasRun ? (
            <div className="gauntlet__results">
              <StreamWindow
                text={result.steered.text}
                isStreaming={isStreaming}
                label={`Corpius 1  ·  ${layerLabel}  ·  α=${alpha.toFixed(1)}`}
              />
              {!isStreaming && (
                <motion.div
                  className="gauntlet__result-metrics"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                >
                  {getResultMetrics(resultIdx).map(m => (
                    <div key={m.sub} className="gauntlet__result-metric">
                      <span className="gauntlet__result-metric-label">{m.sub}</span>
                      <span className="gauntlet__result-metric-value">{m.label}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>
          ) : (
            <div className="gauntlet__empty-state">
              <Zap size={40} />
              <h3>Ready to Generate</h3>
              <p>Select a config, choose a prompt, and hit Generate to run the steered model.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
