/**
 * ProjectCard.tsx — Pollen × Prism card with mini tree thumbnail.
 * Hover lifts subtly; the depth-keyed accent stripe is tinted by node count.
 */

import { Calendar, GitBranch, Trash2 } from 'lucide-react';
import { motion } from 'framer-motion';
import type { Project } from '../types';
import { DEMO_PROJECT_ID } from '../lib/demoProject';
import { depthHue } from '../styles/motion';

interface ProjectCardProps {
  project: Project;
  index: number;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ProjectCard({ project, index, onOpen, onDelete }: ProjectCardProps) {
  const isSampleProject = project.id === DEMO_PROJECT_ID;
  const formattedDate = new Date(project.updatedAt).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  /* Hue chosen by index so the grid reads like a spectrum. */
  const hue = depthHue(index % 5);

  /* Cheap, deterministic constellation thumbnail. */
  const seed = project.id.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const dotCount = Math.min(18, Math.max(4, project.nodeCount || 6));
  const dots = Array.from({ length: dotCount }, (_, i) => {
    const x = ((seed * (i + 7) * 31) % 100);
    const y = (((seed + i) * (i + 13) * 17) % 70) + 10;
    return { x, y, r: 1 + ((seed + i) % 3) * 0.5 };
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: Math.min(index, 5) * 0.05, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -3 }}
      onClick={() => onOpen(project.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onOpen(project.id); }}
      className="group relative rounded-xl border border-[var(--ink-line)] bg-paper-1 hover:bg-paper-2 transition-colors cursor-pointer overflow-hidden focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/50"
      style={{ '--gen-hue': hue, boxShadow: 'var(--shadow-rest)' } as React.CSSProperties}
    >
      {/* Tree thumbnail */}
      <div className="aspect-[16/8] bg-paper border-b border-[var(--ink-line)] relative overflow-hidden">
        <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="absolute inset-0 w-full h-full">
          {dots.map((d, i) => (
            <circle
              key={i}
              cx={d.x}
              cy={d.y}
              r={d.r}
              fill={`oklch(0.62 0.20 ${depthHue((seed + i) % 5)})`}
              opacity={0.55}
            />
          ))}
          {/* connecting strokes */}
          {dots.slice(1).map((d, i) => (
            <line
              key={`l-${i}`}
              x1={dots[0].x}
              y1={dots[0].y}
              x2={d.x}
              y2={d.y}
              stroke={`oklch(0.62 0.20 ${hue})`}
              strokeWidth={0.25}
              opacity={0.3}
            />
          ))}
        </svg>
        <span
          className="absolute top-2 left-2 inline-flex items-center gap-1.5 rounded-full bg-paper-1/85 backdrop-blur px-2 py-0.5 text-[9px] uppercase tracking-[0.18em] text-ink-2"
        >
          <span className="h-1 w-1 rounded-full" style={{ background: 'var(--gen)' }} />
          {project.nodeCount} nodes
        </span>
      </div>

      {/* Body */}
      <div className="p-5">
        <h3 className="font-display text-ink-0 text-[18px] leading-snug tracking-tight mb-1.5" style={{ fontVariationSettings: '"wght" 520, "opsz" 32' }}>
          {project.name}
        </h3>
        {project.description && (
          <p className="text-ink-1 text-[13px] leading-relaxed line-clamp-2 mb-4">{project.description}</p>
        )}
        <div className="flex items-center gap-4 text-[11px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <Calendar size={11} /> {formattedDate}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <GitBranch size={11} /> {project.nodeCount} nodes
          </span>
        </div>
      </div>

      {/* Delete */}
      {!isSampleProject && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (window.confirm(`Delete "${project.name}"? This cannot be undone.`)) {
              onDelete(project.id);
            }
          }}
          title="Delete project"
          aria-label="Delete project"
          className="absolute top-2 right-2 p-1.5 rounded-md bg-paper-1/90 backdrop-blur text-ink-2 hover:bg-paper-2 hover:text-[var(--accent-red)] opacity-0 group-hover:opacity-100 transition-all"
        >
          <Trash2 size={13} />
        </button>
      )}
    </motion.div>
  );
}
