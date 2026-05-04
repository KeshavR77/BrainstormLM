import type { BrainstormPhase, BrainstormEvent } from '../types';

export const IDLE: BrainstormPhase = { status: 'IDLE' };

export function brainstormReducer(
  state: BrainstormPhase,
  event: BrainstormEvent,
): BrainstormPhase {
  if (event.type === 'ERROR') {
    return { status: 'ERROR', message: event.message };
  }

  if (event.type === 'RESET') {
    return IDLE;
  }

  switch (state.status) {
    case 'IDLE':
      if (event.type === 'START_GENERATION') {
        return { status: 'GENERATING_DIVERSE_POOL', progress: 0, rawIdeas: [] };
      }
      return state;

    case 'GENERATING_DIVERSE_POOL':
      if (event.type === 'RAW_IDEA_ARRIVED') {
        return {
          ...state,
          rawIdeas: [...state.rawIdeas, event.idea],
        };
      }
      if (event.type === 'DIVERSE_POOL_DONE') {
        return { status: 'SYNTHESIZING_QUALITY', progress: 0, candidateCount: event.count };
      }
      return state;

    case 'SYNTHESIZING_QUALITY':
      if (event.type === 'SYNTHESIS_PROGRESS') {
        return { ...state, progress: event.progress };
      }
      if (event.type === 'SYNTHESIS_COMPLETE') {
        return { status: 'COMPLETE', finalIdeas: event.ideas };
      }
      return state;

    case 'COMPLETE':
    case 'ERROR':
      return state;
  }
}

export function isLoading(phase: BrainstormPhase): boolean {
  return phase.status === 'GENERATING_DIVERSE_POOL' || phase.status === 'SYNTHESIZING_QUALITY';
}
