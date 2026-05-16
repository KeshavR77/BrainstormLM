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
  ideaId?: string;
  feedbackGiven?: 'up' | 'down';
  isExplored?: boolean;
}

// ── Project Memory ───────────────────────────────────────────────────

export interface ProjectMemoryCounts {
  projectId: string;
  seeded: number;
  generated: number;
  total: number;
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

// ── Brainstorm Phase State Machine ────────────────────────────────────

export interface PartialIdea {
  summary: string;
  content: string;
  temperature?: number;
  creativityScore?: number;
  ideaId?: string;
  pack?: string;
}

export type BrainstormPhase =
  | { status: 'IDLE' }
  | { status: 'GENERATING_DIVERSE_POOL'; progress: number; rawIdeas: PartialIdea[] }
  | { status: 'SYNTHESIZING_QUALITY'; progress: number; candidateCount: number }
  | { status: 'COMPLETE'; finalIdeas: { summary: string; content: string; trajectory_id: string }[] }
  | { status: 'ERROR'; message: string };

export type BrainstormEvent =
  | { type: 'START_GENERATION'; prompt: string }
  | { type: 'RAW_IDEA_ARRIVED'; idea: PartialIdea }
  | { type: 'DIVERSE_POOL_DONE'; count: number }
  | { type: 'SYNTHESIS_PROGRESS'; progress: number }
  | { type: 'SYNTHESIS_COMPLETE'; ideas: { summary: string; content: string; trajectory_id: string; idea_id?: string }[] }
  | { type: 'MEMORY_UPDATED'; counts: ProjectMemoryCounts }
  | { type: 'ERROR'; message: string }
  | { type: 'RESET' };

// ── Steering Telemetry ────────────────────────────────────────────────

export interface SteeringStep {
  step: number;
  creative: number;
  technical: number;
  temperature: number;
}
