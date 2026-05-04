import { describe, it, expect } from 'vitest';
import { brainstormReducer, isLoading, IDLE } from '../lib/brainstormMachine';
import type { BrainstormPhase, PartialIdea } from '../types';

const idea: PartialIdea = { summary: 'Test idea', content: 'Test content' };

describe('brainstormReducer', () => {
  it('starts in IDLE', () => {
    expect(IDLE.status).toBe('IDLE');
  });

  it('IDLE + START_GENERATION => GENERATING_DIVERSE_POOL', () => {
    const next = brainstormReducer(IDLE, { type: 'START_GENERATION', prompt: 'test' });
    expect(next.status).toBe('GENERATING_DIVERSE_POOL');
    if (next.status === 'GENERATING_DIVERSE_POOL') {
      expect(next.rawIdeas).toEqual([]);
      expect(next.progress).toBe(0);
    }
  });

  it('GENERATING_DIVERSE_POOL + RAW_IDEA_ARRIVED => accumulates ideas', () => {
    const generating: BrainstormPhase = {
      status: 'GENERATING_DIVERSE_POOL',
      progress: 0,
      rawIdeas: [],
    };
    const next = brainstormReducer(generating, { type: 'RAW_IDEA_ARRIVED', idea });
    expect(next.status).toBe('GENERATING_DIVERSE_POOL');
    if (next.status === 'GENERATING_DIVERSE_POOL') {
      expect(next.rawIdeas).toHaveLength(1);
      expect(next.rawIdeas[0]).toEqual(idea);
    }
  });

  it('GENERATING_DIVERSE_POOL + DIVERSE_POOL_DONE => SYNTHESIZING_QUALITY', () => {
    const generating: BrainstormPhase = {
      status: 'GENERATING_DIVERSE_POOL',
      progress: 0,
      rawIdeas: [idea],
    };
    const next = brainstormReducer(generating, { type: 'DIVERSE_POOL_DONE', count: 1 });
    expect(next.status).toBe('SYNTHESIZING_QUALITY');
    if (next.status === 'SYNTHESIZING_QUALITY') {
      expect(next.candidateCount).toBe(1);
      expect(next.progress).toBe(0);
    }
  });

  it('SYNTHESIZING_QUALITY + SYNTHESIS_PROGRESS => updates progress', () => {
    const synth: BrainstormPhase = {
      status: 'SYNTHESIZING_QUALITY',
      progress: 0,
      candidateCount: 5,
    };
    const next = brainstormReducer(synth, { type: 'SYNTHESIS_PROGRESS', progress: 0.5 });
    expect(next.status).toBe('SYNTHESIZING_QUALITY');
    if (next.status === 'SYNTHESIZING_QUALITY') {
      expect(next.progress).toBe(0.5);
    }
  });

  it('SYNTHESIZING_QUALITY + SYNTHESIS_COMPLETE => COMPLETE', () => {
    const synth: BrainstormPhase = {
      status: 'SYNTHESIZING_QUALITY',
      progress: 0.8,
      candidateCount: 5,
    };
    const finalIdeas = [{ summary: 'Best', content: 'Best content', trajectory_id: 'traj_1' }];
    const next = brainstormReducer(synth, { type: 'SYNTHESIS_COMPLETE', ideas: finalIdeas });
    expect(next.status).toBe('COMPLETE');
    if (next.status === 'COMPLETE') {
      expect(next.finalIdeas).toEqual(finalIdeas);
    }
  });

  it('any state + ERROR => ERROR', () => {
    const states: BrainstormPhase[] = [
      IDLE,
      { status: 'GENERATING_DIVERSE_POOL', progress: 0, rawIdeas: [] },
      { status: 'SYNTHESIZING_QUALITY', progress: 0, candidateCount: 0 },
    ];
    for (const state of states) {
      const next = brainstormReducer(state, { type: 'ERROR', message: 'fail' });
      expect(next.status).toBe('ERROR');
      if (next.status === 'ERROR') {
        expect(next.message).toBe('fail');
      }
    }
  });

  it('any state + RESET => IDLE', () => {
    const states: BrainstormPhase[] = [
      { status: 'GENERATING_DIVERSE_POOL', progress: 0, rawIdeas: [] },
      { status: 'SYNTHESIZING_QUALITY', progress: 0, candidateCount: 0 },
      { status: 'COMPLETE', finalIdeas: [] },
      { status: 'ERROR', message: 'fail' },
    ];
    for (const state of states) {
      const next = brainstormReducer(state, { type: 'RESET' });
      expect(next.status).toBe('IDLE');
    }
  });

  it('invalid transitions are no-ops', () => {
    // IDLE should ignore RAW_IDEA_ARRIVED
    const next1 = brainstormReducer(IDLE, { type: 'RAW_IDEA_ARRIVED', idea });
    expect(next1).toBe(IDLE);

    // IDLE should ignore SYNTHESIS_COMPLETE
    const next2 = brainstormReducer(IDLE, {
      type: 'SYNTHESIS_COMPLETE',
      ideas: [{ summary: 'x', content: 'x', trajectory_id: 'x' }],
    });
    expect(next2).toBe(IDLE);

    // COMPLETE should ignore START_GENERATION (must RESET first)
    const complete: BrainstormPhase = { status: 'COMPLETE', finalIdeas: [] };
    const next3 = brainstormReducer(complete, { type: 'START_GENERATION', prompt: 'test' });
    expect(next3).toBe(complete);
  });
});

describe('isLoading', () => {
  it('returns false for IDLE', () => {
    expect(isLoading(IDLE)).toBe(false);
  });

  it('returns true for GENERATING_DIVERSE_POOL', () => {
    expect(isLoading({ status: 'GENERATING_DIVERSE_POOL', progress: 0, rawIdeas: [] })).toBe(true);
  });

  it('returns true for SYNTHESIZING_QUALITY', () => {
    expect(isLoading({ status: 'SYNTHESIZING_QUALITY', progress: 0, candidateCount: 0 })).toBe(true);
  });

  it('returns false for COMPLETE', () => {
    expect(isLoading({ status: 'COMPLETE', finalIdeas: [] })).toBe(false);
  });

  it('returns false for ERROR', () => {
    expect(isLoading({ status: 'ERROR', message: 'fail' })).toBe(false);
  });
});
