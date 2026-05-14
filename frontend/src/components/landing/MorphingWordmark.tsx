/**
 * MorphingWordmark — "Conceptial" in Fraunces variable font, with weight + optical-size
 * axes bound to scroll progress. Cheap, distinctive, no animation cost beyond reflow.
 */
import { useScroll, useTransform, motion, useReducedMotion } from 'framer-motion';
import { useRef } from 'react';

interface Props {
  /** Static weight when reduced-motion is preferred */
  staticWeight?: number;
  className?: string;
  children?: React.ReactNode;
}

export function MorphingWordmark({
  staticWeight = 520,
  className = '',
  children = 'Conceptial',
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });

  // Weight 300 → 800 across scroll
  const wght = useTransform(scrollYProgress, [0, 0.5, 1], [320, 620, 820]);
  // Optical size 9 → 144
  const opsz = useTransform(scrollYProgress, [0, 1], [18, 144]);
  // Slight letter-spacing scrub
  const tracking = useTransform(scrollYProgress, [0, 1], [-0.03, 0.01]);

  return (
    <motion.div ref={ref} className={className}>
      <motion.span
        className="font-display text-ink-0 inline-block leading-[0.9] tabular-nums"
        style={{
          fontVariationSettings: reduced
            ? `"wght" ${staticWeight}, "opsz" 72`
            : (useTransform(
                [wght, opsz],
                ([w, o]: number[]) => `"wght" ${w}, "opsz" ${o}`,
              ) as unknown as string),
          letterSpacing: reduced ? '-0.02em' : (tracking as unknown as string),
        }}
      >
        {children}
      </motion.span>
    </motion.div>
  );
}
