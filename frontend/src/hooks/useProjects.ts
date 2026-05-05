/**
 * useProjects.ts — Custom hook for project CRUD with localStorage persistence.
 */

import { useCallback, useEffect, useState } from 'react';
import type { Project } from '../types';
import { DEMO_PROJECT, DEMO_PROJECT_ID } from '../lib/demoProject';

const STORAGE_KEY = 'brainstorm-projects';

function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const projects: Project[] = raw ? JSON.parse(raw) : [];
    const rest = projects.filter((p) => p.id !== DEMO_PROJECT_ID);
    const savedDemo = projects.find((p) => p.id === DEMO_PROJECT_ID);
    const demo = savedDemo
      ? {
          ...DEMO_PROJECT,
          nodeCount: savedDemo.nodeCount,
          updatedAt: savedDemo.updatedAt,
          createdAt: savedDemo.createdAt,
        }
      : { ...DEMO_PROJECT };
    return [demo, ...rest];
  } catch {
    return [{ ...DEMO_PROJECT }];
  }
}

/** Always persist the seeded project first; merge saved row with canonical defaults. */
function saveProjects(projects: Project[]) {
  const demoRow = projects.find((p) => p.id === DEMO_PROJECT_ID);
  const rest = projects.filter((p) => p.id !== DEMO_PROJECT_ID);
  const demo = demoRow
    ? {
        ...DEMO_PROJECT,
        nodeCount: demoRow.nodeCount,
        updatedAt: demoRow.updatedAt,
        createdAt: demoRow.createdAt,
      }
    : DEMO_PROJECT;
  localStorage.setItem(STORAGE_KEY, JSON.stringify([demo, ...rest]));
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
    if (id === DEMO_PROJECT_ID) return;
    setProjects((prev) => prev.filter((p) => p.id !== id));
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
        p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p,
      ),
    );
  }, []);

  return { projects, createProject, deleteProject, getProject, updateProject };
}
