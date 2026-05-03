/**
 * layoutTree.ts — Dagre-based layout for brainstorm tree nodes.
 */

import dagre from 'dagre';
import type { Node as FlowNode, Edge as FlowEdge } from '@xyflow/react';

const NODE_WIDTH = 300;
const NODE_HEIGHT = 140;

/**
 * Apply a top-down dagre layout to the given React Flow nodes and edges.
 * Returns new node objects with `position` set.
 */
export function layoutTree(
  nodes: FlowNode[],
  edges: FlowEdge[],
): FlowNode[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: 'TB',
    ranksep: 100,
    nodesep: 60,
    marginx: 40,
    marginy: 40,
  });

  for (const node of nodes) {
    let w = NODE_WIDTH;
    let h = NODE_HEIGHT;
    if (node.type === 'seed') { w = 360; h = 180; }
    else if (node.type === 'selected') { w = 320; h = 280; }
    g.setNode(node.id, { width: w, height: h });
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    let w = NODE_WIDTH;
    let h = NODE_HEIGHT;
    if (node.type === 'seed') { w = 360; h = 180; }
    else if (node.type === 'selected') { w = 320; h = 280; }
    return {
      ...node,
      position: {
        x: pos.x - w / 2,
        y: pos.y - h / 2,
      },
    };
  });
}
