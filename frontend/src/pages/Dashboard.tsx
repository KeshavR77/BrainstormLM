/**
 * Dashboard.tsx — Conceptial project surface, Pollen × Prism palette.
 *
 * Editorial header, command-bar style controls, project grid with
 * constellation thumbnails. Filter + sort bar will land in Phase 5.
 */

import { useMemo, useState } from 'react';
import { ArrowLeft, Plus, Sparkles, Search } from 'lucide-react';
import { motion } from 'framer-motion';
import { useProjects } from '../hooks/useProjects';
import { ProjectCard } from '../components/ProjectCard';
import { NewProjectModal } from '../components/NewProjectModal';
import { TemplateGallery } from '../components/dashboard/TemplateGallery';
import { Wordmark } from '../components/Logo';
import { staggerContainer } from '../styles/motion';

interface DashboardProps {
  onOpenProject: (id: string) => void;
}

export function Dashboard({ onOpenProject }: DashboardProps) {
  const { projects, createProject, deleteProject } = useProjects();
  const [modalOpen, setModalOpen] = useState(false);
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return projects;
    return projects.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        (p.description?.toLowerCase().includes(q) ?? false),
    );
  }, [query, projects]);

  const handleCreate = (name: string, description: string) => {
    const project = createProject(name, description);
    onOpenProject(project.id);
  };

  return (
    <div className="min-h-screen bg-paper text-ink-0 font-sans antialiased">
      {/* ── Top Bar ──────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 backdrop-blur-md bg-paper/80 border-b border-[var(--ink-line)]">
        <div className="max-w-7xl mx-auto px-6 sm:px-10 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <a
              href="#/"
              className="inline-flex items-center gap-1.5 text-ink-1 hover:text-ink-0 text-sm transition-colors"
            >
              <ArrowLeft size={14} /> Home
            </a>
            <div className="h-4 w-px bg-[var(--ink-line)]" />
            <Wordmark size={19} weight={560} />
            <span className="text-ink-2 text-[12px] hidden sm:inline">· Projects</span>
          </div>

          {/* Search */}
          <div className="hidden md:flex flex-1 max-w-md mx-auto relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-2" />
            <input
              type="search"
              placeholder="Search projects…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[13px] text-ink-0 placeholder:text-ink-3 outline-none focus:border-brand focus:ring-2 focus:ring-brand/15 transition-all"
            />
          </div>

          <div className="flex items-center gap-1 sm:gap-2">
            <a
              href="#/demo"
              className="text-ink-1 hover:text-ink-0 text-[13px] px-3 py-1.5 rounded-md transition-colors hidden sm:inline-block"
            >
              Demo
            </a>
            <button
              onClick={() => setModalOpen(true)}
              className="bg-ink-0 text-paper text-[13px] font-medium px-4 py-2 rounded-full hover:bg-ink-1 transition-colors inline-flex items-center gap-1.5"
            >
              <Plus size={14} /> New project
            </button>
          </div>
        </div>
      </header>

      {/* ── Hero band ────────────────────────────────────────── */}
      <section className="px-6 sm:px-10 pt-12 sm:pt-16 pb-10 max-w-7xl mx-auto">
        <div className="text-ink-2 text-[11px] uppercase tracking-[0.22em] mb-3">Workspace</div>
        <h1
          className="font-display text-ink-0 text-[clamp(34px,5vw,64px)] leading-[1] tracking-[-0.02em] mb-3"
          style={{ fontVariationSettings: '"wght" 380, "opsz" 144' }}
        >
          Where divergent ideas begin.
        </h1>
        <p className="text-ink-1 text-[15px] max-w-xl leading-relaxed">
          Each project is its own idea map. Open one to keep exploring, or start
          fresh — every prompt branches.
        </p>
      </section>

      {/* ── Content ──────────────────────────────────────────── */}
      <main className="px-6 sm:px-10 pb-32 max-w-7xl mx-auto">
        {filtered.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="rounded-2xl border border-dashed border-[var(--ink-line-bold)] bg-paper-1 p-16 text-center max-w-xl mx-auto"
          >
            <div className="mx-auto mb-6 h-12 w-12 rounded-full bg-brand/10 text-brand flex items-center justify-center">
              <Sparkles size={22} />
            </div>
            <h2
              className="font-display text-ink-0 text-[24px] tracking-tight mb-2"
              style={{ fontVariationSettings: '"wght" 520, "opsz" 48' }}
            >
              {query ? 'No projects match' : 'Start your first map'}
            </h2>
            <p className="text-ink-1 text-[14px] mb-6">
              {query
                ? 'Try a different search, or clear the query to see everything.'
                : 'A brainstorm always starts with a single prompt. We\'ll branch from there.'}
            </p>
            {!query && (
              <button
                onClick={() => setModalOpen(true)}
                className="inline-flex items-center gap-2 bg-brand hover:bg-brand-deep text-white text-[14px] font-medium px-5 py-2.5 rounded-full shadow-brand transition-all"
              >
                <Plus size={15} /> Create project
              </button>
            )}
          </motion.div>
        ) : (
          <>
            <div className="flex items-baseline justify-between mb-6">
              <div className="flex items-baseline gap-3">
                <h2
                  className="font-display text-ink-0 text-[22px] tracking-tight"
                  style={{ fontVariationSettings: '"wght" 500, "opsz" 48' }}
                >
                  Your projects
                </h2>
                <span className="text-ink-2 text-[12px] tabular-nums">{filtered.length}</span>
              </div>
            </div>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
            >
              {filtered.map((project, i) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  index={i}
                  onOpen={onOpenProject}
                  onDelete={deleteProject}
                />
              ))}
            </motion.div>
            <TemplateGallery onUseTemplate={handleCreate} />
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
