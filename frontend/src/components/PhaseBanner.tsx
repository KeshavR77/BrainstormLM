import { AnimatePresence, motion } from 'framer-motion';
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import type { BrainstormPhase } from '../types';

interface PhaseBannerProps {
  phase: BrainstormPhase;
}

export function PhaseBanner({ phase }: PhaseBannerProps) {
  return (
    <AnimatePresence mode="wait">
      {phase.status === 'GENERATING_DIVERSE_POOL' && (
        <motion.div
          key="diverse"
          className="phase-banner phase-banner--diverse"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          <span className="phase-banner__dot phase-banner__dot--gold" />
          <Loader2 size={13} className="spin" />
          <span className="phase-banner__text">
            Exploring directions...
            <span className="phase-banner__count">
              {phase.rawIdeas.length} directions found
            </span>
          </span>
        </motion.div>
      )}

      {phase.status === 'SYNTHESIZING_QUALITY' && (
        <motion.div
          key="synth"
          className="phase-banner phase-banner--synth"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          <span className="phase-banner__dot phase-banner__dot--teal" />
          <Loader2 size={13} className="spin" />
          <span className="phase-banner__text">Evaluating quality...</span>
          <div className="phase-banner__progress-track">
            <motion.div
              className="phase-banner__progress-fill"
              initial={{ width: '0%' }}
              animate={{ width: `${phase.progress * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </motion.div>
      )}

      {phase.status === 'COMPLETE' && (
        <motion.div
          key="complete"
          className="phase-banner phase-banner--complete"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          <CheckCircle2 size={13} />
          <span className="phase-banner__text">
            {phase.finalIdeas.length} ideas ready
          </span>
        </motion.div>
      )}

      {phase.status === 'ERROR' && (
        <motion.div
          key="error"
          className="phase-banner phase-banner--error"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          <AlertCircle size={13} />
          <span className="phase-banner__text">{phase.message}</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
