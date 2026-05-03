/**
 * useBrainstorm.ts — Manages the brainstorm tree state for a single project.
 *
 * Handles node creation, idea generation, selection, editing, and expansion.
 * All state is persisted to localStorage keyed by project ID.
 *
 * RLHF feedback loop:
 * - Each generated idea carries a trajectoryId from the backend.
 * - "Choose This" auto-fires thumbs-up for the chosen idea and thumbs-down
 *   for all siblings, creating DPO preference pairs automatically.
 * - submitNodeFeedback / submitCritique allow explicit feedback from the UI.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { BrainstormNode, BrainstormEdge, BrainstormTree } from '../types';
import { generateIdeas, submitFeedback } from '../lib/api';

function storageKey(projectId: string) {
  return `brainstorm-tree-${projectId}`;
}

function loadTree(projectId: string): BrainstormTree {
  try {
    const raw = localStorage.getItem(storageKey(projectId));
    return raw ? JSON.parse(raw) : { nodes: {}, edges: [], rootId: null };
  } catch {
    return { nodes: {}, edges: [], rootId: null };
  }
}

function saveTree(projectId: string, tree: BrainstormTree) {
  localStorage.setItem(storageKey(projectId), JSON.stringify(tree));
}

export function useBrainstorm(projectId: string) {
  const [tree, setTree] = useState<BrainstormTree>(() => loadTree(projectId));
  const [loading, setLoading] = useState(false);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const treeRef = useRef(tree);
  treeRef.current = tree;

  // Persist tree whenever it changes
  useEffect(() => {
    saveTree(projectId, tree);
  }, [projectId, tree]);

  // Re-load tree when projectId changes
  useEffect(() => {
    setTree(loadTree(projectId));
    setActiveNodeId(null);
  }, [projectId]);

  // ── Submit seed ────────────────────────────────────────────────

  const submitSeed = useCallback(
    async (text: string) => {
      const seedId = crypto.randomUUID();
      const seed: BrainstormNode = {
        id: seedId,
        parentId: null,
        type: 'seed',
        content: text.trim(),
        summary: text.trim().slice(0, 60) + (text.length > 60 ? '…' : ''),
        depth: 0,
        createdAt: new Date().toISOString(),
        isExpanded: false,
        children: [],
      };

      setTree({ nodes: { [seedId]: seed }, edges: [], rootId: seedId });
      setActiveNodeId(seedId);

      // Auto-generate first set of ideas
      await expandNodeInner(seedId, undefined, {
        nodes: { [seedId]: seed },
        edges: [],
        rootId: seedId,
      });
    },
    [projectId],
  );

  // ── Expand a node (generate children) ─────────────────────────

  const expandNodeInner = useCallback(
    async (nodeId: string, userPrompt?: string, currentTree?: BrainstormTree) => {
      const t = currentTree ?? treeRef.current;
      const node = t.nodes[nodeId];
      if (!node || node.isExpanded) return;

      setLoading(true);
      try {
        const ideas = await generateIdeas(node.content, userPrompt);
        const newNodes: Record<string, BrainstormNode> = {};
        const newEdges: BrainstormEdge[] = [];
        const childIds: string[] = [];

        for (const idea of ideas) {
          const childId = crypto.randomUUID();
          childIds.push(childId);

          newNodes[childId] = {
            id: childId,
            parentId: nodeId,
            type: 'idea',
            content: idea.content,
            summary: idea.summary,
            depth: node.depth + 1,
            createdAt: new Date().toISOString(),
            isExpanded: false,
            children: [],
            trajectoryId: idea.trajectory_id,
            feedbackGiven: undefined,
          };

          newEdges.push({
            id: `e-${nodeId}-${childId}`,
            source: nodeId,
            target: childId,
          });
        }

        setTree((prev) => ({
          ...prev,
          nodes: {
            ...prev.nodes,
            [nodeId]: { ...prev.nodes[nodeId], isExpanded: true, children: childIds },
            ...newNodes,
          },
          edges: [...prev.edges, ...newEdges],
        }));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const expandNode = useCallback(
    async (nodeId: string) => {
      await expandNodeInner(nodeId);
    },
    [expandNodeInner],
  );

  // ── Expand with user prompt ───────────────────────────────────

  const expandWithPrompt = useCallback(
    async (nodeId: string, userPrompt: string) => {
      await expandNodeInner(nodeId, userPrompt);
    },
    [expandNodeInner],
  );

  // ── Select an idea — fires DPO preference pairs automatically ─

  const selectIdea = useCallback(
    (nodeId: string) => {
      const node = treeRef.current.nodes[nodeId];
      if (!node || node.type === 'seed') return;

      // Fire thumbs-up for chosen idea
      if (node.trajectoryId) {
        submitFeedback(node.trajectoryId, 'up');
      }

      // Fire thumbs-down for all siblings that haven't been explicitly rated
      const parentNode = node.parentId ? treeRef.current.nodes[node.parentId] : null;
      if (parentNode) {
        for (const sibId of parentNode.children) {
          if (sibId !== nodeId) {
            const sib = treeRef.current.nodes[sibId];
            if (sib?.trajectoryId && !sib.feedbackGiven) {
              submitFeedback(sib.trajectoryId, 'down');
            }
          }
        }
      }

      // Update state: mark chosen as selected, siblings as rejected
      setTree((prev) => {
        const updated = { ...prev, nodes: { ...prev.nodes } };
        // Mark chosen node selected
        updated.nodes[nodeId] = {
          ...updated.nodes[nodeId],
          type: 'selected',
          feedbackGiven: 'up',
        };
        // Mark siblings as having received down feedback
        if (parentNode) {
          for (const sibId of parentNode.children) {
            if (sibId !== nodeId && updated.nodes[sibId] && !updated.nodes[sibId].feedbackGiven) {
              updated.nodes[sibId] = { ...updated.nodes[sibId], feedbackGiven: 'down' };
            }
          }
        }
        return updated;
      });

      setActiveNodeId(nodeId);
    },
    [],
  );

  // ── Manual feedback (explicit thumb press on a node) ──────────

  const submitNodeFeedback = useCallback(
    (nodeId: string, thumbs: 'up' | 'down') => {
      const node = treeRef.current.nodes[nodeId];
      if (!node?.trajectoryId || node.feedbackGiven) return;

      submitFeedback(node.trajectoryId, thumbs);

      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [nodeId]: { ...prev.nodes[nodeId], feedbackGiven: thumbs },
        },
      }));
    },
    [],
  );

  // ── Submit a critique note for a selected node ────────────────

  const submitCritique = useCallback(
    (nodeId: string, critique: string) => {
      const node = treeRef.current.nodes[nodeId];
      if (!node?.trajectoryId || !critique.trim()) return;
      submitFeedback(node.trajectoryId, 'up', undefined, critique.trim());
    },
    [],
  );

  // ── Edit a node's content ─────────────────────────────────────

  const editNode = useCallback(
    (nodeId: string, content: string, summary?: string) => {
      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [nodeId]: {
            ...prev.nodes[nodeId],
            content,
            summary: summary ?? content.slice(0, 60) + (content.length > 60 ? '…' : ''),
          },
        },
      }));
    },
    [],
  );

  // ── Add a custom idea as a child ──────────────────────────────

  const addCustomIdea = useCallback(
    (parentId: string, content: string, summary: string) => {
      const parent = treeRef.current.nodes[parentId];
      if (!parent) return;

      const childId = crypto.randomUUID();

      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [parentId]: {
            ...prev.nodes[parentId],
            isExpanded: true,
            children: [...prev.nodes[parentId].children, childId],
          },
          [childId]: {
            id: childId,
            parentId,
            type: 'idea' as const,
            content: content.trim(),
            summary: summary.trim() || content.trim().slice(0, 40),
            depth: parent.depth + 1,
            createdAt: new Date().toISOString(),
            isExpanded: false,
            children: [],
          },
        },
        edges: [
          ...prev.edges,
          { id: `e-${parentId}-${childId}`, source: parentId, target: childId },
        ],
      }));
    },
    [],
  );

  // ── Computed values ───────────────────────────────────────────

  const nodeCount = Object.keys(tree.nodes).length;
  const maxDepth = Object.values(tree.nodes).reduce(
    (max, n) => Math.max(max, n.depth),
    0,
  );

  return {
    tree,
    loading,
    activeNodeId,
    setActiveNodeId,
    submitSeed,
    expandNode,
    expandWithPrompt,
    selectIdea,
    editNode,
    addCustomIdea,
    submitNodeFeedback,
    submitCritique,
    nodeCount,
    maxDepth,
    hasRoot: tree.rootId !== null,
  };
}
