/**
 * Canvas.tsx — The brainstorming workspace with React Flow canvas.
 */

import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  BackgroundVariant,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type Node as FlowNode,
  type Edge as FlowEdge,
  type OnNodesChange,
  type OnEdgesChange,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Sparkles } from 'lucide-react';

import { nodeTypes } from '../components/nodes';
import { CanvasHeader } from '../components/CanvasHeader';
import { useBrainstorm } from '../hooks/useBrainstorm';
import { useProjects } from '../hooks/useProjects';
import { layoutTree } from '../layoutTree';

interface CanvasPageProps {
  projectId: string;
  onBack: () => void;
}

function CanvasInner({ projectId, onBack }: CanvasPageProps) {
  const {
    tree,
    loading,
    submitSeed,
    expandNode,
    expandWithPrompt,
    selectIdea,
    editNode,
    addCustomIdea,
    nodeCount,
    maxDepth,
    hasRoot,
  } = useBrainstorm(projectId);

  const { getProject, updateProject } = useProjects();
  const project = getProject(projectId);
  const { fitView } = useReactFlow();

  const [flowNodes, setFlowNodes] = useState<FlowNode[]>([]);
  const [flowEdges, setFlowEdges] = useState<FlowEdge[]>([]);
  const prevNodeCount = useRef(0);

  // ── Update project node count ────────────────────────────────
  useEffect(() => {
    if (nodeCount > 0) {
      updateProject(projectId, { nodeCount });
    }
  }, [nodeCount, projectId, updateProject]);

  // ── Convert brainstorm tree → React Flow nodes/edges ─────────

  const { builtNodes, builtEdges } = useMemo(() => {
    const fNodes: FlowNode[] = [];
    const fEdges: FlowEdge[] = [];

    for (const node of Object.values(tree.nodes)) {
      const base = { id: node.id, position: { x: 0, y: 0 } };

      switch (node.type) {
        case 'seed':
          fNodes.push({
            ...base,
            type: 'seed',
            data: {
              content: node.content,
              submitted: true,
              onSubmit: submitSeed,
            },
          });
          break;

        case 'idea':
          fNodes.push({
            ...base,
            type: 'idea',
            data: {
              summary: node.summary,
              content: node.content,
              depth: node.depth,
              isExpanded: node.isExpanded,
              hasChildren: node.children.length > 0,
              onSelect: selectIdea,
              onExpand: expandNode,
              onEdit: editNode,
              nodeId: node.id,
            },
          });
          break;

        case 'selected':
          fNodes.push({
            ...base,
            type: 'selected',
            data: {
              summary: node.summary,
              content: node.content,
              depth: node.depth,
              isExpanded: node.isExpanded,
              onExpandWithPrompt: expandWithPrompt,
              onAutoExpand: expandNode,
              onAddCustom: addCustomIdea,
              nodeId: node.id,
            },
          });
          break;
      }
    }

    for (const edge of tree.edges) {
      fEdges.push({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        animated: true,
      });
    }

    return { builtNodes: fNodes, builtEdges: fEdges };
  }, [tree, submitSeed, selectIdea, expandNode, expandWithPrompt, editNode, addCustomIdea]);

  // ── Apply layout ─────────────────────────────────────────────

  useEffect(() => {
    if (builtNodes.length > 0) {
      const laid = layoutTree(builtNodes, builtEdges);
      setFlowNodes(laid);
      setFlowEdges(builtEdges);
    }
  }, [builtNodes, builtEdges]);

  // ── Auto-fit view ────────────────────────────────────────────

  useEffect(() => {
    if (flowNodes.length > 0 && flowNodes.length !== prevNodeCount.current) {
      prevNodeCount.current = flowNodes.length;
      setTimeout(() => fitView({ padding: 0.3, duration: 500 }), 150);
    }
  }, [flowNodes, fitView]);

  // ── Allow dragging ───────────────────────────────────────────

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setFlowNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setFlowEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );

  // ── Initial state: show seed input ───────────────────────────

  const displayNodes: FlowNode[] = !hasRoot
    ? [
        {
          id: 'initial-seed',
          type: 'seed',
          position: { x: 0, y: 0 },
          data: {
            // Pre-fill with the project description when the user
            // kicked off from the landing-page hero.
            content: project?.description ?? '',
            submitted: false,
            onSubmit: submitSeed,
          },
        },
      ]
    : flowNodes;

  return (
    <div className="canvas-page">
      <CanvasHeader
        projectName={project?.name ?? 'Untitled'}
        nodeCount={nodeCount}
        maxDepth={maxDepth}
        loading={loading}
        onBack={onBack}
      />

      <div className="canvas-viewport">
        <ReactFlow
          nodes={displayNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          minZoom={0.1}
          maxZoom={1.8}
          fitView
          fitViewOptions={{ padding: 0.5 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={24}
            size={1}
            color="rgba(201, 165, 92, 0.10)"
          />
          <Controls position="bottom-left" showInteractive={false} />
        </ReactFlow>

        {/* Hint overlay when canvas has no root */}
        {!hasRoot && (
          <div className="canvas-hint animate-fade-in">
            <Sparkles size={16} />
            Type your idea above and press{' '}
            <kbd className="canvas-hint__kbd">⌘ Enter</kbd> to start
          </div>
        )}
      </div>
    </div>
  );
}

// Wrap with ReactFlowProvider
export function Canvas(props: CanvasPageProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
