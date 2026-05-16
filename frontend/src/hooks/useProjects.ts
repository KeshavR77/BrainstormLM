/**
 * useProjects.ts — Project CRUD with localStorage persistence.
 *
 * Always merges the canonical demo projects (onboarding demo + the three
 * creative-ops-themed seeded projects) into whatever the user has in localStorage,
 * so the dashboard never opens to an empty state and the seeded demos can't
 * be accidentally deleted.
 */

import { useCallback, useEffect, useState } from 'react';
import type { Project } from '../types';
import {
  DEMO_PROJECT,
  DEMO_PROJECT_ID,
  PROTECTED_PROJECT_IDS,
  SEEDED_DEMO_PROJECTS,
} from '../lib/demoProject';

const STORAGE_KEY = 'brainstorm-projects';

const CANONICAL_DEMOS: Project[] = [DEMO_PROJECT, ...SEEDED_DEMO_PROJECTS];

function mergeWithDemos(saved: Project[]): Project[] {
  // For each canonical demo, prefer the saved row if one exists (so
  // nodeCount and updatedAt persist) but fall back to canonical defaults
  // for name / description / etc.
  const byId = new Map(saved.map((p) => [p.id, p]));
  const merged: Project[] = [];

  for (const canon of CANONICAL_DEMOS) {
    const savedRow = byId.get(canon.id);
    if (savedRow) {
      merged.push({
        ...canon,
        nodeCount: savedRow.nodeCount,
        updatedAt: savedRow.updatedAt,
        createdAt: savedRow.createdAt,
      });
    } else {
      merged.push(canon);
    }
  }

  const rest = saved.filter((p) => !PROTECTED_PROJECT_IDS.has(p.id));
  return [...merged, ...rest];
}

function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const saved: Project[] = raw ? JSON.parse(raw) : [];
    return mergeWithDemos(saved);
  } catch {
    return [...CANONICAL_DEMOS];
  }
}

function saveProjects(projects: Project[]) {
  // Persist saved-row metadata for canonical demos AND any user projects.
  const out: Project[] = [];
  const byId = new Map(projects.map((p) => [p.id, p]));
  for (const canon of CANONICAL_DEMOS) {
    const row = byId.get(canon.id);
    if (row) {
      out.push({
        ...canon,
        nodeCount: row.nodeCount,
        updatedAt: row.updatedAt,
        createdAt: row.createdAt,
      });
    }
  }
  for (const p of projects) {
    if (!PROTECTED_PROJECT_IDS.has(p.id)) out.push(p);
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
}

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>(loadProjects);

  useEffect(() => {
    saveProjects(projects);
  }, [projects]);

  const createProject = useCallback((name: string, description: string): Project => {
    const project: Project = {
      id: crypto.randomUUID(),
      name: name.trim(),
      description: description.trim(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      nodeCount: 0,
    };
    setProjects((prev) => [project, ...prev]);
    return project;
  }, []);

  const deleteProject = useCallback((id: string) => {
    if (PROTECTED_PROJECT_IDS.has(id)) return;
    setProjects((prev) => prev.filter((p) => p.id !== id));
    localStorage.removeItem(`brainstorm-tree-${id}`);
  }, []);

  const getProject = useCallback(
    (id: string): Project | undefined => {
      return projects.find((p) => p.id === id);
    },
    [projects],
  );

  const updateProject = useCallback(
    (id: string, updates: Partial<Pick<Project, 'name' | 'description' | 'nodeCount'>>) => {
      setProjects((prev) =>
        prev.map((p) =>
          p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p,
        ),
      );
    },
    [],
  );

  return { projects, createProject, deleteProject, getProject, updateProject };
}

export { DEMO_PROJECT_ID };
