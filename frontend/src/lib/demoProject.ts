import type { BrainstormTree, Project } from '../types';

export const DEMO_PROJECT_ID = 'demo-video';
const DEMO_TIMESTAMP = '2026-05-04T00:00:00.000Z';

export const DEMO_PROJECT: Project = {
  id: DEMO_PROJECT_ID,
  name: 'Onboarding redesign',
  description: 'Idea map for a faster first-session aha moment.',
  createdAt: DEMO_TIMESTAMP,
  updatedAt: DEMO_TIMESTAMP,
  nodeCount: 7,
};

// ── Seeded demo projects for creative-ops audiences ─────────────────────────
//
// These three projects are pre-loaded with hand-curated seed ideas in the
// server-side IdeaMemoryStore (run scripts/seed_idea_memory.py --all to
// populate). On the canvas they open with "Memory: N seeded · 0 generated"
// and immediately demonstrate the project-scoped memory feature.

const AIR_TIMESTAMP = '2026-05-16T00:00:00.000Z';

export const SEEDED_DEMO_SKINCARE_ID = 'demo-skincare-identity';
export const SEEDED_DEMO_FOOTWEAR_ID = 'demo-footwear-campaign';
export const SEEDED_DEMO_OATMILK_ID = 'demo-oatmilk-packaging';

export const SEEDED_DEMO_PROJECTS: Project[] = [
  {
    id: SEEDED_DEMO_SKINCARE_ID,
    name: 'Cypress & Salt — Brand Identity',
    description:
      'A DTC skincare brand. 12 visual-identity directions already explored. Generate the next set — they will not repeat what is already in memory.',
    createdAt: AIR_TIMESTAMP,
    updatedAt: AIR_TIMESTAMP,
    nodeCount: 0,
  },
  {
    id: SEEDED_DEMO_FOOTWEAR_ID,
    name: 'Bondi Trail — Q3 Campaign',
    description:
      'A sustainable trail-runner launch. 8 campaign concept directions already explored. Generate fresh angles that do not retread the existing pool.',
    createdAt: AIR_TIMESTAMP,
    updatedAt: AIR_TIMESTAMP,
    nodeCount: 0,
  },
  {
    id: SEEDED_DEMO_OATMILK_ID,
    name: 'Larder — Oat Milk Packaging',
    description:
      'A premium oat milk reformulation. 8 packaging directions already explored. Generate new ones that do not overlap the seeded set.',
    createdAt: AIR_TIMESTAMP,
    updatedAt: AIR_TIMESTAMP,
    nodeCount: 0,
  },
];

export const SEEDED_DEMO_IDS = new Set(SEEDED_DEMO_PROJECTS.map((p) => p.id));

/** Project IDs that should never be user-deletable from the dashboard. */
export const PROTECTED_PROJECT_IDS = new Set<string>([
  DEMO_PROJECT_ID,
  ...SEEDED_DEMO_IDS,
]);

export const DEMO_TREE: BrainstormTree = {
  rootId: 'demo-seed',
  nodes: {
    'demo-seed': {
      id: 'demo-seed',
      parentId: null,
      type: 'seed',
      content: 'How should a design team redesign onboarding so new users reach an aha moment in under 60 seconds?',
      summary: 'Redesign onboarding for a sub-60-second aha moment',
      depth: 0,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: true,
      children: ['demo-instant-payoff', 'demo-progressive-trust', 'demo-social-proof'],
    },
    'demo-instant-payoff': {
      id: 'demo-instant-payoff',
      parentId: 'demo-seed',
      type: 'selected',
      content: 'Start from the user goal, not the product tour. Ask one role-specific question, then generate a useful artifact immediately so the user feels progress before setup begins.',
      summary: 'Instant personal payoff',
      depth: 1,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: true,
      children: ['demo-role-task', 'demo-sandbox', 'demo-receipt'],
      feedbackGiven: 'up',
    },
    'demo-progressive-trust': {
      id: 'demo-progressive-trust',
      parentId: 'demo-seed',
      type: 'idea',
      content: 'Delay integrations, imports, and account setup until after the product has shown value. Let users explore with reversible low-commitment steps first.',
      summary: 'Progressive trust',
      depth: 1,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: false,
      children: [],
      feedbackGiven: 'down',
    },
    'demo-social-proof': {
      id: 'demo-social-proof',
      parentId: 'demo-seed',
      type: 'idea',
      content: 'Open with examples from similar teams so users can recognize their destination before learning the interface.',
      summary: 'Social proof moment',
      depth: 1,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: false,
      children: [],
      feedbackGiven: 'down',
    },
    'demo-role-task': {
      id: 'demo-role-task',
      parentId: 'demo-instant-payoff',
      type: 'idea',
      content: 'Route designers, founders, and PMs into different first tasks. Each path should produce a concrete asset: a flow audit, launch brief, or product narrative.',
      summary: 'Role-based first task',
      depth: 2,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: false,
      children: [],
    },
    'demo-sandbox': {
      id: 'demo-sandbox',
      parentId: 'demo-instant-payoff',
      type: 'idea',
      content: 'Use sample data to let users experience the finished workflow before they connect tools, import files, or invite a teammate.',
      summary: 'Sample data sandbox',
      depth: 2,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: false,
      children: [],
    },
    'demo-receipt': {
      id: 'demo-receipt',
      parentId: 'demo-instant-payoff',
      type: 'idea',
      content: 'End onboarding with an outcome receipt: what the user created, why it matters, and the next action they can share with their team.',
      summary: 'Outcome receipt',
      depth: 2,
      createdAt: DEMO_TIMESTAMP,
      isExpanded: false,
      children: [],
    },
  },
  edges: [
    { id: 'e-demo-seed-demo-instant-payoff', source: 'demo-seed', target: 'demo-instant-payoff' },
    { id: 'e-demo-seed-demo-progressive-trust', source: 'demo-seed', target: 'demo-progressive-trust' },
    { id: 'e-demo-seed-demo-social-proof', source: 'demo-seed', target: 'demo-social-proof' },
    { id: 'e-demo-instant-payoff-demo-role-task', source: 'demo-instant-payoff', target: 'demo-role-task' },
    { id: 'e-demo-instant-payoff-demo-sandbox', source: 'demo-instant-payoff', target: 'demo-sandbox' },
    { id: 'e-demo-instant-payoff-demo-receipt', source: 'demo-instant-payoff', target: 'demo-receipt' },
  ],
};
