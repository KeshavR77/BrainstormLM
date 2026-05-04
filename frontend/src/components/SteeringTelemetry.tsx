import { AnimatePresence, motion } from 'framer-motion';
import type { BrainstormPhase } from '../types';
import scheduleData from '../mocks/steering-schedule.json';

interface Props {
  phase: BrainstormPhase;
}

const W = 240;
const H = 96;
const PAD = 12;
const INNER_W = W - PAD * 2;
const INNER_H = H - PAD * 2 - 16; // 16px for label row

const schedule = scheduleData.schedule;
const N = schedule.length;

function toX(step: number) {
  return PAD + (step / (N - 1)) * INNER_W;
}

function toY(val: number) {
  return PAD + 16 + (1 - val) * INNER_H;
}

function buildPath(key: 'creative' | 'technical') {
  return schedule
    .map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.step).toFixed(1)} ${toY(s[key]).toFixed(1)}`)
    .join(' ');
}

const creativePath = buildPath('creative');
const technicalPath = buildPath('technical');

export function SteeringTelemetry({ phase }: Props) {
  const active =
    phase.status === 'GENERATING_DIVERSE_POOL' || phase.status === 'SYNTHESIZING_QUALITY';

  const currentStep =
    phase.status === 'GENERATING_DIVERSE_POOL'
      ? Math.min(phase.rawIdeas.length, N - 1)
      : phase.status === 'SYNTHESIZING_QUALITY'
      ? N - 1
      : 0;

  const curX = toX(currentStep);
  const curStep = schedule[Math.min(currentStep, N - 1)];

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          className="steering-telemetry"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.3 }}
        >
          <div className="steering-telemetry__header">
            <span className="steering-telemetry__label">Steering Schedule</span>
            <span className="steering-telemetry__temp">
              τ = {curStep.temperature.toFixed(2)}
            </span>
          </div>

          <svg width={W} height={H} className="steering-telemetry__svg">
            {/* Creative vector — gold */}
            <path
              d={creativePath}
              fill="none"
              stroke="rgba(201,165,92,0.7)"
              strokeWidth={1.5}
              strokeLinecap="round"
            />
            {/* Technical vector — teal */}
            <path
              d={technicalPath}
              fill="none"
              stroke="rgba(94,234,212,0.7)"
              strokeWidth={1.5}
              strokeLinecap="round"
            />
            {/* Current step cursor */}
            <motion.line
              x1={curX}
              x2={curX}
              y1={PAD + 16}
              y2={PAD + 16 + INNER_H}
              stroke="rgba(255,255,255,0.25)"
              strokeWidth={1}
              strokeDasharray="3 3"
              animate={{ x1: curX, x2: curX }}
              transition={{ duration: 0.3 }}
            />
          </svg>

          <div className="steering-telemetry__readouts">
            <span className="steering-telemetry__readout steering-telemetry__readout--gold">
              creative {(curStep.creative * 100).toFixed(0)}%
            </span>
            <span className="steering-telemetry__readout steering-telemetry__readout--teal">
              technical {(curStep.technical * 100).toFixed(0)}%
            </span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
