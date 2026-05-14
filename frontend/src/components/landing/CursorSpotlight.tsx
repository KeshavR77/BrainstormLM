/**
 * CursorSpotlight — a soft radial gradient mask following the pointer,
 * laid over a faint dot grid. Replaces particles. Pointer-coarse devices
 * fall back to a centered ambient glow.
 */
import { useEffect, useRef, useState } from 'react';

interface Props {
  /** Dot grid color (CSS), default uses --ink-line-soft */
  dotColor?: string;
  /** Spotlight radius in px */
  radius?: number;
  /** Spotlight color (CSS); defaults to --gen-soft */
  spotColor?: string;
  className?: string;
}

export function CursorSpotlight({
  dotColor = 'rgba(22,19,24,0.035)',
  radius = 360,
  spotColor = 'oklch(0.92 0.06 var(--gen-hue) / 0.38)',
  className = '',
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onMove = (e: PointerEvent) => {
      if (!active) setActive(true);
      const r = el.getBoundingClientRect();
      el.style.setProperty('--mx', `${e.clientX - r.left}px`);
      el.style.setProperty('--my', `${e.clientY - r.top}px`);
    };
    const onLeave = () => setActive(false);
    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('pointerleave', onLeave);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerleave', onLeave);
    };
  }, [active]);

  return (
    <>
      {/* Persistent faint dot grid — full-bleed, no mask */}
      <div
        aria-hidden="true"
        className={`pointer-events-none fixed inset-0 ${className}`}
        style={{
          backgroundImage: `radial-gradient(circle at center, ${dotColor} 1px, transparent 1.4px)`,
          backgroundSize: '36px 36px',
          zIndex: 0,
        }}
      />
      {/* Spotlight only renders when cursor is active over the page */}
      <div
        ref={ref}
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 overflow-hidden transition-opacity duration-500"
        style={{ opacity: active ? 1 : 0, zIndex: 0 }}
      >
        <div
          className="absolute"
          style={{
            left: 'var(--mx, 50%)',
            top: 'var(--my, 50%)',
            width: radius * 2,
            height: radius * 2,
            transform: 'translate(-50%, -50%)',
            background: `radial-gradient(circle, ${spotColor} 0%, transparent 65%)`,
            filter: 'blur(28px)',
          }}
        />
      </div>
    </>
  );
}
