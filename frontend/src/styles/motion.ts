/**
 * motion.ts — Shared Framer Motion variants.
 *
 * Editorial pacing: deliberate, slightly slower than default, clip-path
 * reveals over opacity. Each variant uses the same easing curve so the
 * page feels like one orchestrated entrance.
 */
import type { Variants, Transition } from 'framer-motion';

export const easeOut:    Transition['ease'] = [0.22, 1, 0.36, 1];

/** Stagger container with slow editorial cadence. */
export const staggerContainer: Variants = {
  hidden:  {},
  visible: { transition: { staggerChildren: 0.09, delayChildren: 0.1 } },
};

/** Tighter stagger for chips/badges. */
export const staggerFast: Variants = {
  hidden:  {},
  visible: { transition: { staggerChildren: 0.055, delayChildren: 0.08 } },
};

/** Line reveal — slides up from below (mask via parent overflow:hidden). */
export const linesReveal: Variants = {
  hidden:  { y: '110%', opacity: 0 },
  visible: { y: '0%', opacity: 1, transition: { duration: 0.95, ease: easeOut } },
};

/** Fade + rise for body sections. */
export const fadeRise: Variants = {
  hidden:  { opacity: 0, y: 32 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.75, ease: easeOut } },
};

/** Soft pop for chips. */
export const popIn: Variants = {
  hidden:  { opacity: 0, scale: 0.94, y: 8 },
  visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.55, ease: easeOut } },
};

/** Depth-keyed OKLCH hue ramp for tree nodes. */
export const depthHue = (depth: number): number => {
  const ramp = [280, 320, 25, 140, 200, 260];
  return ramp[depth % ramp.length];
};
