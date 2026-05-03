/**
 * useProjects.ts — Custom hook for project CRUD with localStorage persistence.
 */

import { useCallback, useEffect, useState } from 'react';
import type { Project } from '../types';

const STORAGE_KEY = 'brainstorm-projects';

function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveProjects(projects: Project[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
}

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>(loadProjects);

  // Persist whenever projects change
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
    setProjects((prev) => prev.filter((p) => p.id !== id));
    // Also remove the brainstorm tree for this project
    localStorage.removeItem(`brainstorm-tree-${id}`);
  }, []);

  const getProject = useCallback(
    (id: string): Project | undefined => {
      return projects.find((p) => p.id === id);
    },
    [projects],
  );

  const updateProject = useCallback((id: string, updates: Partial<Pick<Project, 'name' | 'description' | 'nodeCount'>>) => {
    setProjects((prev) =>
      prev.map((p) =>
        p.id === id
          ? { ...p, ...updates, updatedAt: new Date().toISOString() }
          : p,
      ),
    );
  }, []);

  return { projects, createProject, deleteProject, getProject, updateProject };
}
