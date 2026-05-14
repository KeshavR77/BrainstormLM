/**
 * Logo — Conceptial mark.
 *
 * A small idea-tree glyph: one root node fanning into three branches.
 * Pure SVG, scales with `size` prop. Defaults to a 22px ink mark with a
 * magenta brand root. Use this everywhere the brand appears.
 */
interface Props {
  size?: number;
  /** Mark color (defaults to var(--ink-0)) */
  color?: string;
  /** Root dot color (defaults to var(--brand)) */
  accent?: string;
  className?: string;
}

export function Logo({ size = 22, color = 'var(--ink-0)', accent = 'var(--brand)', className = '' }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      role="img"
      aria-label="Conceptial"
      className={className}
    >
      {/* Vertical trunk from root → fork */}
      <path d="M12 6.4 V 11.2" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
      {/* Left branch curve */}
      <path
        d="M12 11.2 C 12 14, 6.5 13, 6.5 16.8"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" fill="none"
      />
      {/* Center stem */}
      <path d="M12 11.2 V 16.8" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
      {/* Right branch curve */}
      <path
        d="M12 11.2 C 12 14, 17.5 13, 17.5 16.8"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" fill="none"
      />
      {/* Leaf nodes — small, balanced */}
      <circle cx="6.5"  cy="17.4" r="1.4" fill={color} />
      <circle cx="12"   cy="17.4" r="1.4" fill={color} />
      <circle cx="17.5" cy="17.4" r="1.4" fill={color} />
      {/* Root — accent magenta dot, balanced size */}
      <circle cx="12" cy="5.6" r="1.9" fill={accent} />
    </svg>
  );
}

/** Wordmark variant: mark + "Conceptial" in Fraunces. */
export function Wordmark({
  size = 22,
  weight = 580,
  className = '',
}: {
  size?: number;
  weight?: number;
  className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <Logo size={size} />
      <span
        className="font-display text-ink-0 tracking-tight"
        style={{ fontSize: size * 0.82, fontVariationSettings: `"wght" ${weight}, "opsz" 32` }}
      >
        Conceptial
      </span>
    </span>
  );
}
