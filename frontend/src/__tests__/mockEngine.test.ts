import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mockGeneratePhased } from '../lib/mockEngine';
import type { BrainstormEvent } from '../types';

describe('mockGeneratePhased', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('same prompt always produces same pack (deterministic)', async () => {
    const events1: BrainstormEvent[] = [];
    const events2: BrainstormEvent[] = [];

    const p1 = mockGeneratePhased('test prompt', undefined, (e) => events1.push(e));
    const p2 = mockGeneratePhased('test prompt', undefined, (e) => events2.push(e));

    await vi.runAllTimersAsync();
    const result1 = await p1;
    const result2 = await p2;

    expect(result1.map((r) => r.summary)).toEqual(result2.map((r) => r.summary));
  });

  it('different prompts can produce different packs', async () => {
    const events1: BrainstormEvent[] = [];
    const events2: BrainstormEvent[] = [];

    const p1 = mockGeneratePhased('alpha prompt', undefined, (e) => events1.push(e));
    const p2 = mockGeneratePhased('completely different topic', undefined, (e) => events2.push(e));

    await vi.runAllTimersAsync();
    const result1 = await p1;
    const result2 = await p2;

    expect(result1.length).toBe(4);
    expect(result2.length).toBe(4);
  });

  it('dispatches events in correct order', async () => {
    const events: BrainstormEvent[] = [];
    const p = mockGeneratePhased('test order', undefined, (e) => events.push(e));

    await vi.runAllTimersAsync();
    await p;

    const types = events.map((e) => e.type);

    // Should start with RAW_IDEA_ARRIVED events
    const firstRaw = types.indexOf('RAW_IDEA_ARRIVED');
    expect(firstRaw).toBeGreaterThanOrEqual(0);

    // Then DIVERSE_POOL_DONE
    const poolDone = types.indexOf('DIVERSE_POOL_DONE');
    expect(poolDone).toBeGreaterThan(firstRaw);

    // Then SYNTHESIS_PROGRESS events
    const synthProgress = types.indexOf('SYNTHESIS_PROGRESS');
    expect(synthProgress).toBeGreaterThan(poolDone);

    // Then SYNTHESIS_COMPLETE
    const synthComplete = types.indexOf('SYNTHESIS_COMPLETE');
    expect(synthComplete).toBeGreaterThan(synthProgress);
  });

  it('dispatches 10+ RAW_IDEA_ARRIVED events', async () => {
    const events: BrainstormEvent[] = [];
    const p = mockGeneratePhased('count test', undefined, (e) => events.push(e));

    await vi.runAllTimersAsync();
    await p;

    const rawCount = events.filter((e) => e.type === 'RAW_IDEA_ARRIVED').length;
    expect(rawCount).toBeGreaterThanOrEqual(10);
  });

  it('returns exactly 4 synthesized ideas', async () => {
    const events: BrainstormEvent[] = [];
    const p = mockGeneratePhased('synth test', undefined, (e) => events.push(e));

    await vi.runAllTimersAsync();
    const result = await p;

    expect(result).toHaveLength(4);
    for (const idea of result) {
      expect(idea).toHaveProperty('summary');
      expect(idea).toHaveProperty('content');
      expect(idea).toHaveProperty('trajectory_id');
      expect(idea.trajectory_id).toBeTruthy();
    }
  });

  it('SYNTHESIS_COMPLETE ideas match returned value', async () => {
    const events: BrainstormEvent[] = [];
    const p = mockGeneratePhased('match test', undefined, (e) => events.push(e));

    await vi.runAllTimersAsync();
    const result = await p;

    const completeEvent = events.find((e) => e.type === 'SYNTHESIS_COMPLETE');
    expect(completeEvent).toBeDefined();
    if (completeEvent && completeEvent.type === 'SYNTHESIS_COMPLETE') {
      expect(completeEvent.ideas).toEqual(result);
    }
  });
});
