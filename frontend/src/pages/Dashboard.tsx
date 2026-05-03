/**
 * Dashboard.tsx — Landing page showing all brainstorm projects.
 */

import { useState } from 'react';
import { Plus, Sparkles, Zap } from 'lucide-react';
import { useProjects } from '../hooks/useProjects';
import { ProjectCard } from '../components/ProjectCard';
import { NewProjectModal } from '../components/NewProjectModal';

interface DashboardProps {
  onOpenProject: (id: string) => void;
}

export function Dashboard({ onOpenProject }: DashboardProps) {
  const { projects, createProject, deleteProject } = useProjects();
  const [modalOpen, setModalOpen] = useState(false);

  const handleCreate = (name: string, description: string) => {
    const project = createProject(name, description);
    onOpenProject(project.id);
  };

  return (
    <div className="dashboard">
      <div className="dashboard__bg" />

      {/* ── Header ────────────────────────────────────────────── */}
      <header className="dashboard__header">
        <div className="dashboard__brand">
          <div className="dashboard__logo">
            <Zap size={16} />
          </div>
          <div>
            <h1 className="dashboard__title">Conceptial</h1>
            <p className="dashboard__subtitle">Where divergent ideas begin</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <a href="#/" className="btn btn-ghost" aria-label="Home">
            ← Home
          </a>
          <button className="btn btn-primary" onClick={() => setModalOpen(true)}>
            <Plus size={16} />
            New Project
          </button>
        </div>
      </header>

      {/* ── Content ───────────────────────────────────────────── */}
      <main className="dashboard__content">
        {projects.length === 0 ? (
          <div className="dashboard__empty animate-fade-in">
            <div className="dashboard__empty-icon">
              <Sparkles size={40} />
            </div>
            <h2>No projects yet</h2>
            <p>Create your first brainstorming project to get started.</p>
            <button
              className="btn btn-primary btn-lg"
              onClick={() => setModalOpen(true)}
            >
              <Plus size={18} />
              Create Your First Project
            </button>
          </div>
        ) : (
          <>
            <div className="dashboard__section-header">
              <h2 className="dashboard__section-title">Your Projects</h2>
              <span className="badge badge-subtle">{projects.length}</span>
            </div>
            <div className="dashboard__grid">
              {projects.map((project, i) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  index={i}
                  onOpen={onOpenProject}
                  onDelete={deleteProject}
                />
              ))}
            </div>
          </>
        )}
      </main>

      <NewProjectModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onCreate={handleCreate}
      />
    </div>
  );
}
