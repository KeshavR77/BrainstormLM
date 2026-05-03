/**
 * nodes/index.ts — Node type registry for React Flow.
 */

import { SeedNode } from './SeedNode';
import { IdeaNode } from './IdeaNode';
import { SelectedIdeaNode } from './SelectedIdeaNode';

export const nodeTypes = {
  seed: SeedNode,
  idea: IdeaNode,
  selected: SelectedIdeaNode,
};
