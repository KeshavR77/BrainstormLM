/**
 * types.ts — Core type definitions for the Brainstorm app.
 */

// ── Project ─────────────────────────────────────────────────────────

export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
  nodeCount: number;
}

// ── Brainstorm Tree ─────────────────────────────────────────────────

export type BrainstormNodeType = 'seed' | 'idea' | 'selected';

export interface BrainstormNode {
  id: string;
  parentId: string | null;
  type: BrainstormNodeType;
  content: string;
  summary: string;
  depth: number;
  createdAt: string;
  isExpanded: boolean;
  children: string[];
  trajectoryId?: string;
  feedbackGiven?: 'up' | 'down';
}

export interface BrainstormEdge {
  id: string;
  source: string;
  target: string;
}

export interface BrainstormTree {
  nodes: Record<string, BrainstormNode>;
  edges: BrainstormEdge[];
  rootId: string | null;
}
