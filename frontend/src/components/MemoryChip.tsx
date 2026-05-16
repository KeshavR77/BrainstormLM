/**
 * MemoryChip — floating UI chip showing per-project memory state.
 *
 * Reads the seeded / generated split from the SSE memory event stream and
 * the initial /v1/memory/count endpoint. Sits in the bottom-right of the
 * canvas next to the SteeringTelemetry SVG. The point of this chip is to
 * make the project-scoped memory feature visible — without it, the anti-mode
 * exclusion is invisible technology. With it, the audience sees the number
 * tick up and understands that the model is accumulating context.
 */

import { motion } from 'framer-motion';
import { Database } from 'lucide-react';
import type { ProjectMemoryCounts } from '../types';

interface Props {
  counts: ProjectMemoryCounts;
  onReset?: () => void;
  variant?: 'compact' | 'expanded';
}

export function MemoryChip({ counts, onReset, variant = 'expanded' }: Props) {
  if (counts.total === 0) {
    return (
      <motion.div
        className="memory-chip memory-chip--empty"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        title="Project memory is empty. As you generate ideas, this chip tracks every direction you've explored — and prevents the model from suggesting them again."
      >
        <Database size={13} strokeWidth={1.6} />
        <span>Memory: empty</span>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="memory-chip"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      title={`This project has seen ${counts.total} ideas. The model will not re-suggest them.`}
    >
      <Database size={13} strokeWidth={1.6} />
      {variant === 'expanded' ? (
        <span className="memory-chip__readout">
          Memory:&nbsp;
          {counts.seeded > 0 && (
            <>
              <strong>{counts.seeded}</strong>&nbsp;seeded
            </>
          )}
          {counts.seeded > 0 && counts.generated > 0 && <span className="memory-chip__sep">·</span>}
          {counts.generated > 0 && (
            <>
              <strong>{counts.generated}</strong>&nbsp;explored
            </>
          )}
        </span>
      ) : (
        <span className="memory-chip__readout">
          <strong>{counts.total}</strong>&nbsp;in&nbsp;memory
        </span>
      )}
      {onReset && counts.total > 0 && (
        <button
          type="button"
          className="memory-chip__reset"
          onClick={onReset}
          title="Clear this project's memory and start fresh"
        >
          reset
        </button>
      )}
    </motion.div>
  );
}
