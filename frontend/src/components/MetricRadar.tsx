import { motion } from 'framer-motion';

interface Props {
  diversity: number;
  coherence: number;
  novelty: number;
  feasibility: number;
  size?: number;
}

const LABELS = ['Diversity', 'Coherence', 'Novelty', 'Feasibility'];

export function MetricRadar({
  diversity,
  coherence,
  novelty,
  feasibility,
  size = 200,
}: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.38;
  const values = [diversity, coherence, novelty, feasibility];
  const n = values.length;

  const angleOf = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;

  const pointAt = (i: number, val: number) => {
    const a = angleOf(i);
    return { x: cx + Math.cos(a) * r * val, y: cy + Math.sin(a) * r * val };
  };

  const polygon = values.map((v, i) => {
    const p = pointAt(i, v);
    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
  }).join(' ');

  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <div className="radar" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {rings.map(rv => (
          <polygon
            key={rv}
            points={Array.from({ length: n }, (_, i) => {
              const p = pointAt(i, rv);
              return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
            }).join(' ')}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={1}
          />
        ))}

        {Array.from({ length: n }, (_, i) => {
          const p = pointAt(i, 1);
          return (
            <line
              key={i}
              x1={cx}
              y1={cy}
              x2={p.x}
              y2={p.y}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
          );
        })}

        <motion.polygon
          points={polygon}
          fill="rgba(201,165,92,0.12)"
          stroke="rgba(201,165,92,0.7)"
          strokeWidth={1.5}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        />

        {values.map((v, i) => {
          const p = pointAt(i, v);
          return (
            <motion.circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={3}
              fill="#c9a55c"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, delay: 0.3 + i * 0.1 }}
            />
          );
        })}

        {LABELS.map((label, i) => {
          const p = pointAt(i, 1.22);
          return (
            <text
              key={label}
              x={p.x}
              y={p.y}
              textAnchor="middle"
              dominantBaseline="central"
              fill="rgba(237,235,232,0.45)"
              fontSize={10}
              fontFamily="var(--font-sans)"
            >
              {label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
