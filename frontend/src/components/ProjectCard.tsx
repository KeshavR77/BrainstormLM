/**
 * ProjectCard.tsx — Glass card component for the dashboard project grid.
 */

import { Calendar, GitBranch, Trash2 } from 'lucide-react';
import type { Project } from '../types';

interface ProjectCardProps {
  project: Project;
  index: number;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

export function ProjectCard({ project, index, onOpen, onDelete }: ProjectCardProps) {
  const formattedDate = new Date(project.updatedAt).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div
      className={`project-card glass-card animate-fade-in animate-stagger-${Math.min(index + 1, 6)}`}
      onClick={() => onOpen(project.id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onOpen(project.id);
      }}
    >
      <div className="project-card__gradient" />
      <div className="project-card__content">
        <h3 className="project-card__name">{project.name}</h3>
        {project.description && (
          <p className="project-card__desc">{project.description}</p>
        )}
        <div className="project-card__meta">
          <span className="project-card__meta-item">
            <Calendar size={12} />
            {formattedDate}
          </span>
          <span className="project-card__meta-item">
            <GitBranch size={12} />
            {project.nodeCount} nodes
          </span>
        </div>
      </div>
      <button
        className="project-card__delete"
        onClick={(e) => {
          e.stopPropagation();
          if (window.confirm(`Delete "${project.name}"? This cannot be undone.`)) {
            onDelete(project.id);
          }
        }}
        title="Delete project"
        aria-label="Delete project"
      >
        <Trash2 size={14} />
      </button>
    </div>
  );
}
