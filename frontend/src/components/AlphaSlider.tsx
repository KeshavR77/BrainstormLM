import { motion } from 'framer-motion';

interface Props {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function AlphaSlider({ value, onChange, min = 0, max = 3, step = 0.1 }: Props) {
  const pct = ((value - min) / (max - min)) * 100;

  const label =
    value < 0.8 ? 'Conservative' :
    value < 1.5 ? 'Moderate' :
    value < 2.2 ? 'Creative' :
    'Maximum Divergence';

  const color =
    value < 0.8 ? 'rgba(94,234,212,0.8)' :
    value < 1.5 ? 'rgba(201,165,92,0.8)' :
    value < 2.2 ? 'rgba(212,185,120,0.9)' :
    'rgba(248,113,113,0.9)';

  return (
    <div className="alpha-slider">
      <div className="alpha-slider__header">
        <span className="alpha-slider__label">
          Steering Strength <span className="alpha-slider__greek">&alpha;</span>
        </span>
        <motion.span
          className="alpha-slider__value"
          style={{ color }}
          key={value.toFixed(1)}
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.15 }}
        >
          {value.toFixed(1)}
        </motion.span>
      </div>

      <div className="alpha-slider__track-wrapper">
        <div className="alpha-slider__gradient-bar">
          <div className="alpha-slider__fill" style={{ width: `${pct}%` }} />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="alpha-slider__input"
          aria-label="Alpha steering strength"
        />
      </div>

      <div className="alpha-slider__footer">
        <span className="alpha-slider__hint" style={{ color }}>{label}</span>
        <span className="alpha-slider__range">{min} &mdash; {max}</span>
      </div>
    </div>
  );
}
