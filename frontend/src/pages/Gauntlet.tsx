import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Layers, Zap, Play } from 'lucide-react';
import { AlphaSlider } from '../components/AlphaSlider';
import { StreamWindow } from '../components/StreamWindow';
import { MetricRadar } from '../components/MetricRadar';
import mockData from '../mocks/mock_results.json';

const presets = mockData.gauntlet_presets;
const comparisons = mockData.comparisons;

interface GauntletProps {
  onBack: () => void;
}

type LayerMode = 'mid' | 'late';

export function Gauntlet({ onBack }: GauntletProps) {
  const [alpha, setAlpha] = useState(1.8);
  const [layer, setLayer] = useState<LayerMode>('mid');
  const [prompt, setPrompt] = useState(comparisons[0].prompt);
  const [isStreaming, setIsStreaming] = useState(false);
  const [resultIdx, setResultIdx] = useState(0);
  const [hasRun, setHasRun] = useState(false);

  const result = comparisons[resultIdx];

  const handleGenerate = useCallback(() => {
    setIsStreaming(true);
    setHasRun(true);
    setResultIdx(Math.floor(Math.random() * comparisons.length));
    setTimeout(() => setIsStreaming(false), 2800);
  }, []);

  const predictedMetrics = {
    diversity: Math.min(1, 0.3 + alpha * 0.28),
    coherence: Math.max(0.2, 1 - alpha * 0.12),
    novelty: Math.min(1, 0.2 + alpha * 0.32),
    feasibility: Math.max(0.3, 0.95 - alpha * 0.18),
  };

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
            <span className="badge badge-brand">LIVE DEMO</span>
          </div>
        </div>
        <div className="gauntlet__header-right">
          <span className="gauntlet__model-badge">
            Qwen2.5-14B-Instruct
          </span>
        </div>
      </header>

      <main className="gauntlet__main">
        <div className="gauntlet__controls">
          <div className="gauntlet__panel">
            <h3 className="gauntlet__panel-title">Steering Controls</h3>

            <AlphaSlider value={alpha} onChange={setAlpha} />

            <div className="gauntlet__layer-select">
              <label className="gauntlet__field-label">
                <Layers size={13} /> Injection Layer
              </label>
              <div className="gauntlet__layer-toggle">
                <button
                  className={`gauntlet__layer-btn ${layer === 'mid' ? 'gauntlet__layer-btn--active' : ''}`}
                  onClick={() => setLayer('mid')}
                >
                  Mid-Layer (L14)
                </button>
                <button
                  className={`gauntlet__layer-btn ${layer === 'late' ? 'gauntlet__layer-btn--active' : ''}`}
                  onClick={() => setLayer('late')}
                >
                  Late-Layer (L18)
                </button>
              </div>
            </div>

            <div className="gauntlet__presets">
              <label className="gauntlet__field-label">Presets</label>
              <div className="gauntlet__preset-grid">
                {presets.map(p => (
                  <button
                    key={p.label}
                    className={`gauntlet__preset ${
                      alpha === p.alpha && layer === p.layer ? 'gauntlet__preset--active' : ''
                    }`}
                    onClick={() => {
                      setAlpha(p.alpha);
                      setLayer(p.layer as LayerMode);
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="gauntlet__panel">
            <h3 className="gauntlet__panel-title">Predicted Metrics</h3>
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
                <span className="gauntlet__metric-val gauntlet__metric-val--gold">
                  {(predictedMetrics.diversity * 100).toFixed(0)}%
                </span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Coherence</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--teal">
                  {(predictedMetrics.coherence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Novelty</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--gold">
                  {(predictedMetrics.novelty * 100).toFixed(0)}%
                </span>
              </div>
              <div className="gauntlet__metric-item">
                <span>Feasibility</span>
                <span className="gauntlet__metric-val gauntlet__metric-val--teal">
                  {(predictedMetrics.feasibility * 100).toFixed(0)}%
                </span>
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
                label={`Steered Output  |  α=${alpha.toFixed(1)}  |  ${layer === 'mid' ? 'L14' : 'L18'}`}
              />

              {!isStreaming && (
                <motion.div
                  className="gauntlet__result-metrics"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                >
                  <div className="gauntlet__result-metric">
                    <span className="gauntlet__result-metric-label">Vendi Score</span>
                    <span className="gauntlet__result-metric-value">{result.metrics.vendi_score.toFixed(2)}</span>
                  </div>
                  <div className="gauntlet__result-metric">
                    <span className="gauntlet__result-metric-label">PRISM Coherence</span>
                    <span className="gauntlet__result-metric-value">{(result.metrics.prism_coherence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="gauntlet__result-metric">
                    <span className="gauntlet__result-metric-label">Novelty</span>
                    <span className="gauntlet__result-metric-value">{(result.metrics.novelty * 100).toFixed(0)}%</span>
                  </div>
                  <div className="gauntlet__result-metric">
                    <span className="gauntlet__result-metric-label">Feasibility</span>
                    <span className="gauntlet__result-metric-value">{(result.metrics.feasibility * 100).toFixed(0)}%</span>
                  </div>
                </motion.div>
              )}
            </div>
          ) : (
            <div className="gauntlet__empty-state">
              <Zap size={40} />
              <h3>Ready to Generate</h3>
              <p>Adjust the steering controls, enter a prompt, and hit Generate to see the steered model in action.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
