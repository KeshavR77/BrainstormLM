/**
 * IntroSplash — first-impression intro that plays once per session.
 *
 * Timeline (~2.9s):
 *   0.0–0.6s   root pop + glow
 *   0.45–1.4s  trunk + branches draw
 *   1.1–1.5s   leaves pop in
 *   1.45–2.1s  Conceptial wordmark fades up
 *   2.2–2.9s   whole overlay fades out
 */
import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface Props { onDone: () => void }

export function IntroSplash({ onDone }: Props) {
  const [opacity, setOpacity] = useState(1);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    const t1 = window.setTimeout(() => setOpacity(0), 2200);
    const t2 = window.setTimeout(() => onDoneRef.current(), 2900);
    return () => { window.clearTimeout(t1); window.clearTimeout(t2); };
  }, []);

  return (
    <div
      className="fixed inset-0 flex items-center justify-center bg-paper transition-opacity duration-[700ms]"
      style={{ zIndex: 200, opacity, pointerEvents: opacity === 0 ? 'none' : 'auto' }}
    >
      {/* Subtle dot grid */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none opacity-40"
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(22,19,24,0.05) 1px, transparent 1.4px)',
          backgroundSize: '34px 34px',
        }}
      />
      {/* Vignette */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(circle at 50% 48%, transparent 0%, transparent 35%, rgba(244, 239, 230, 0.7) 80%)',
        }}
      />

      <div className="relative flex flex-col items-center gap-6">
        <AnimatedLogo />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.45, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-center gap-2"
        >
          <div
            className="font-display text-ink-0 text-[30px] sm:text-[34px] tracking-[-0.02em] leading-none"
            style={{ fontVariationSettings: '"wght" 560, "opsz" 48' }}
          >
            Conceptial
          </div>
          <div className="text-[10.5px] uppercase tracking-[0.28em] text-ink-2">
            non-linear reasoning engine
          </div>
        </motion.div>
      </div>
    </div>
  );
}

function AnimatedLogo() {
  const ease = [0.22, 1, 0.36, 1] as const;
  const spring = [0.34, 1.56, 0.64, 1] as const;

  return (
    <motion.svg
      width={96}
      height={96}
      viewBox="0 0 24 24"
      fill="none"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease }}
    >
      <defs>
        <radialGradient id="splashRootGlow">
          <stop offset="0%"   stopColor="var(--brand)" stopOpacity="0.4" />
          <stop offset="60%"  stopColor="var(--brand)" stopOpacity="0.05" />
          <stop offset="100%" stopColor="var(--brand)" stopOpacity="0" />
        </radialGradient>
      </defs>
      <motion.circle
        cx={12} cy={5.6} r={5}
        fill="url(#splashRootGlow)"
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 0.9, scale: 1 }}
        transition={{ duration: 0.6, ease }}
      />
      <motion.path
        d="M12 6.4 V 11.2"
        stroke="var(--ink-0)" strokeWidth={1.5} strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.35, delay: 0.45, ease }}
      />
      <motion.path
        d="M12 11.2 C 12 14, 6.5 13, 6.5 16.8"
        stroke="var(--ink-0)" strokeWidth={1.5} strokeLinecap="round" fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.55, delay: 0.7, ease }}
      />
      <motion.path
        d="M12 11.2 V 16.8"
        stroke="var(--ink-0)" strokeWidth={1.5} strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.55, delay: 0.78, ease }}
      />
      <motion.path
        d="M12 11.2 C 12 14, 17.5 13, 17.5 16.8"
        stroke="var(--ink-0)" strokeWidth={1.5} strokeLinecap="round" fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.55, delay: 0.86, ease }}
      />
      <motion.circle
        cx={6.5} cy={17.4} r={1.4} fill="var(--ink-0)"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.4, delay: 1.05, ease: spring }}
      />
      <motion.circle
        cx={12} cy={17.4} r={1.4} fill="var(--ink-0)"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.4, delay: 1.14, ease: spring }}
      />
      <motion.circle
        cx={17.5} cy={17.4} r={1.4} fill="var(--ink-0)"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.4, delay: 1.23, ease: spring }}
      />
      <motion.circle
        cx={12} cy={5.6} r={1.9} fill="var(--brand)"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2, ease: spring }}
      />
    </motion.svg>
  );
}
