import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import {
  AIMessage,
  AIMessageChunk,
  ChatMessage,
  ChatMessageChunk,
  FunctionMessage,
  FunctionMessageChunk,
  HumanMessage,
  HumanMessageChunk,
  RemoveMessage,
  SystemMessage,
  SystemMessageChunk,
  ToolMessage,
  ToolMessageChunk,
  isBaseMessage,
} from '@langchain/core/messages';
import { MemorySaver, GraphRecursionError } from '@langchain/langgraph';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { isRunStepResumeState } from '@/tools/runStepResume';
import { Run, CHECKPOINT_CONTINUATION_VERSION } from '@/run';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { HookRegistry } from '@/hooks/HookRegistry';
import { FakeChatModel } from '@/llm/fake';
import { askUserQuestion } from '@/hitl';

/** Real Run/LangGraph/MemorySaver contracts; only the provider is deterministic.
 * A fresh Run below proves object reconstruction, not operating-system durability.
 */
const THREAD = 'checkpoint-contract-thread';
const RUN = 'checkpoint-contract-run';
const config = (): t.RunStreamConfig => ({
  version: 'v2',
  configurable: { thread_id: THREAD },
});
const input = (): t.IState => ({
  messages: [
    new HumanMessage({ content: 'original prompt', id: 'source-human' }),
  ],
});

class SiblingModel extends FakeChatModel {
  readonly visits: string[] = [];
  private readonly agentsByRun = new Map<string, string>();

  constructor(private readonly before: (agent: string) => Promise<void>) {
    super({ responses: ['sibling result'] });
    this.callbacks = [
      {
        handleChatModelStart: (
          _model,
          _messages,
          runId,
          _parentId,
          _extra,
          _tags,
          metadata
        ) => {
          if (typeof metadata?.activeAgentId === 'string') {
            this.agentsByRun.set(runId, metadata.activeAgentId);
          }
        },
      },
    ];
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    manager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const agent =
      manager == null ? undefined : this.agentsByRun.get(manager.runId);
    if (agent == null || agent === '')
      throw new Error('Actual agent metadata missing');
    this.visits.push(agent);
    await this.before(agent);
    yield* super._streamResponseChunks(messages, options, manager);
  }
}

async function createSiblings(saver: MemorySaver, model: SiblingModel) {
  const run = await Run.create<t.IState>({
    runId: RUN,
    graphConfig: {
      type: 'multi-agent',
      agents: ['A', 'B'].map((agentId) => ({
        agentId,
        instructions: `IDENTITY:${agentId}`,
        provider: Providers.OPENAI,
        clientOptions: {
          model: 'synthetic-model',
          apiKey: 'synthetic-unused-key',
          streaming: true,
        },
        maxContextTokens: 8000,
      })),
      edges: [],
      compileOptions: { checkpointer: saver },
    },
    checkpointPolicy: { threadId: THREAD, durability: 'sync' },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) throw new Error('Expected actual multi-agent graph');
  run.Graph.overrideModel = model;
  return run;
}

async function createRun(options: {
  saver?: MemorySaver;
  runId?: string;
  policy?: t.RunConfig['checkpointPolicy'];
  model?: FakeChatModel;
  tools?: t.AgentInputs['graphTools'];
  fallback?: boolean;
  inspectCheckpointInput?: t.RunConfig['inspectCheckpointInput'];
  hookText?: string;
}) {
  const hooks: string[] = [];
  const steps: Array<{ id: string; index: number }> = [];
  const completions: string[] = [];
  const registry = new HookRegistry();
  registry.register('RunStart', {
    hooks: [
      async () => {
        hooks.push('start');
        return options.hookText != null && options.hookText.length > 0
          ? { additionalContext: options.hookText } : {};
      },
    ],
  });
  registry.register('UserPromptSubmit', {
    hooks: [
      async () => {
        hooks.push('prompt');
        return {};
      },
    ],
  });
  const run = await Run.create<t.IState>({
    runId: options.runId ?? RUN,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'continuation-agent',
          provider: Providers.OPENAI,
          clientOptions: {
            model: 'synthetic-model',
            apiKey: 'synthetic-unused-key',
            streaming: true,
          },
          maxContextTokens: 8000,
          instructions: 'Deterministic checkpoint fixture.',
          graphTools: options.tools,
        },
      ],
      ...(options.saver && { compileOptions: { checkpointer: options.saver } }),
    },
    checkpointPolicy: options.policy,
    inspectCheckpointInput: options.inspectCheckpointInput,
    humanInTheLoop: { enabled: options.fallback === true },
    returnContent: true,
    skipCleanup: true,
    hooks: registry,
    customHandlers: {
      [GraphEvents.ON_RUN_STEP]: {
        handle: (_event, data) => {
          if (
            data == null ||
            !('id' in data) ||
            !('index' in data) ||
            typeof data.id !== 'string' ||
            typeof data.index !== 'number'
          ) {
            throw new Error('Invalid actual step event');
          }
          steps.push({ id: data.id, index: data.index });
        },
      },
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (_event, data) => {
          const result =
            data != null && 'result' in data ? data.result : undefined;
          if (
            result != null &&
            typeof result === 'object' &&
            'id' in result &&
            typeof result.id === 'string'
          ) {
            completions.push(result.id);
          }
        },
      },
    },
  });
  const model =
    options.model ?? new FakeChatModel({ responses: ['continued answer'] });
  if (!run.Graph) throw new Error('Expected the actual initialized graph');
  run.Graph.overrideModel = model;
  return {
    run,
    modelCalls: jest.spyOn(model, '_streamResponseChunks'),
    hooks,
    steps,
    completions,
  };
}

async function checkpoint(saver: MemorySaver): Promise<t.RunStreamConfig> {
  const tuple = await saver.getTuple(config());
  if (
    tuple == null ||
    typeof tuple.config.configurable?.checkpoint_id !== 'string' ||
    tuple.config.configurable.checkpoint_id === ''
  )
    throw new Error('Real checkpoint missing');
  return { version: 'v2', configurable: { ...tuple.config.configurable } };
}

async function pending() {
  const saver = new MemorySaver();
  const effects: string[] = [];
  const echo = tool(
    async ({ text }) => {
      effects.push(text);
      return `echo:${text}`;
    },
    {
      name: 'echo_checkpoint',
      description: 'Record the synthetic tool effect.',
      schema: z.object({ text: z.string() }),
    }
  );
  const policy: NonNullable<t.RunConfig['checkpointPolicy']> = {
    threadId: THREAD,
    durability: 'sync',
  };
  const first = await createRun({
    saver,
    policy,
    tools: [echo],
    model: new FakeChatModel({
      responses: ['requesting echo'],
      toolCalls: [
        {
          name: 'echo_checkpoint',
          args: { text: 'once' },
          id: 'original-tool-call',
          type: 'tool_call',
        },
      ],
    }),
  });
  await expect(
    first.run.processStream(input(), { ...config(), recursionLimit: 2 })
  ).rejects.toBeInstanceOf(GraphRecursionError);
  const exact = await checkpoint(saver);
  let open: t.RunStep | undefined;
  for await (const tuple of saver.list(config())) {
    const state = tuple.checkpoint.channel_values.runStepState;
    if (isRunStepResumeState(state)) {
      open = state.steps.find(
        (entry) => entry.step.type === StepTypes.TOOL_CALLS
      )?.step;
      if (open) break;
    }
  }
  if (!open) throw new Error('Expected an actual persisted tool step');
  // The real recursion boundary follows the tool, before the next model turn.
  expect(effects).toEqual(['once']);
  return { saver, effects, echo, policy, first, exact, open };
}

afterEach(() => jest.restoreAllMocks());

describe('public native checkpoint continuation', () => {
  it('continues a real pending graph in a fresh Run with the original output and step identities', async () => {
    expect(CHECKPOINT_CONTINUATION_VERSION).toBe(1);
    const fixture = await pending();
    const second = await createRun({
      saver: fixture.saver,
      policy: fixture.policy,
      tools: [fixture.echo],
    });
    const result = await second.run.continueFromCheckpoint(fixture.exact);
    expect(fixture.effects).toEqual(['once']);
    expect(second.modelCalls).toHaveBeenCalledTimes(1);
    expect(second.hooks).toEqual([]);
    expect(second.run.id).toBe(RUN);
    expect(JSON.stringify(result)).toContain('continued answer');
    const latest = await fixture.saver.getTuple(config());
    expect(latest?.checkpoint.channel_values.runStepState).toMatchObject({
      runId: RUN,
    });
    expect(
      fixture.first.completions.filter((id) => id === fixture.open.id)
    ).toHaveLength(1);
    expect(second.completions).not.toContain(fixture.open.id);
    expect(second.steps.length).toBeGreaterThan(0);
    for (const step of second.steps) {
      expect(step.index).toBeGreaterThan(fixture.open.index);
      expect(fixture.first.steps.map(({ id }) => id)).not.toContain(step.id);
    }
    const messages = latest?.checkpoint.channel_values.messages;
    if (!Array.isArray(messages))
      throw new Error('Expected actual checkpoint messages');
    expect(
      messages
        .filter(isBaseMessage)
        .filter((message) => message.getType() === 'human')
        .map((message) => message.id)
    ).toEqual(['source-human']);
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          tool_call_id: 'original-tool-call',
          content: 'echo:once',
        }),
      ])
    );
  });

  it.each(['none', 'implicit-hitl'] as const)(
    'rejects %s persistence without invoking the model',
    async (mode) => {
      const fixture = await createRun({ fallback: mode === 'implicit-hitl' });
      await expect(
        fixture.run.continueFromCheckpoint({
          ...config(),
          configurable: {
            thread_id: THREAD,
            checkpoint_id: 'missing',
          },
        })
      ).rejects.toThrow(/checkpoint|saver|persist/i);
      expect(fixture.modelCalls).not.toHaveBeenCalled();
      expect(fixture.hooks).toEqual([]);
    }
  );

  it.each([
    'missing-thread',
    'missing-checkpoint',
    'wrong-thread',
    'unknown-checkpoint',
    'namespace',
    'foreign-run',
    'missing-run',
  ] as const)(
    'rejects %s identity before model, hooks, or persistence writes',
    async (reason) => {
      const fixture = await pending();
      const options = {
        ...fixture.exact,
        configurable: { ...fixture.exact.configurable },
      };
      if (reason === 'missing-thread') delete options.configurable.thread_id;
      if (reason === 'missing-checkpoint')
        delete options.configurable.checkpoint_id;
      if (reason === 'wrong-thread')
        options.configurable.thread_id = 'foreign-thread';
      if (reason === 'unknown-checkpoint')
        options.configurable.checkpoint_id = 'unknown-checkpoint';
      if (reason === 'namespace')
        options.configurable.checkpoint_ns = 'foreign-namespace';
      if (reason === 'missing-run') {
        const tuple = await fixture.saver.getTuple(fixture.exact);
        if (!tuple?.metadata)
          throw new Error('Expected real checkpoint metadata');
        const state = tuple.checkpoint.channel_values.runStepState;
        if (!isRunStepResumeState(state))
          throw new Error('Expected actual native sidecars');
        const { runId: _runId, ...unbound } = state;
        await fixture.saver.put(
          tuple.config,
          {
            ...tuple.checkpoint,
            channel_values: {
              ...tuple.checkpoint.channel_values,
              runStepState: unbound,
            },
          },
          tuple.metadata
        );
      }
      const second = await createRun({
        saver: fixture.saver,
        policy: fixture.policy,
        tools: [fixture.echo],
        ...(reason === 'foreign-run' && { runId: 'foreign-run' }),
      });
      const writes = jest.spyOn(fixture.saver, 'put');
      const pendingWrites = jest.spyOn(fixture.saver, 'putWrites');
      await expect(second.run.continueFromCheckpoint(options)).rejects.toThrow(
        /checkpoint|thread|namespace|run/i
      );
      expect(second.modelCalls).not.toHaveBeenCalled();
      expect(second.hooks).toEqual([]);
      expect(writes).not.toHaveBeenCalled();
      expect(pendingWrites).not.toHaveBeenCalled();
      expect(fixture.effects).toEqual(['once']);
    }
  );

  it('rejects both an old checkpoint and the completed latest checkpoint without starting a new run', async () => {
    const fixture = await pending();
    const second = await createRun({
      saver: fixture.saver,
      policy: fixture.policy,
      tools: [fixture.echo],
    });
    await second.run.continueFromCheckpoint(fixture.exact);
    const latest = await checkpoint(fixture.saver);
    expect(latest.configurable?.checkpoint_id).not.toBe(
      fixture.exact.configurable?.checkpoint_id
    );
    const third = await createRun({
      saver: fixture.saver,
      policy: fixture.policy,
      tools: [fixture.echo],
    });
    const writes = jest.spyOn(fixture.saver, 'put');
    await expect(
      third.run.continueFromCheckpoint(fixture.exact)
    ).rejects.toThrow(/checkpoint|stale|latest/i);
    await expect(third.run.continueFromCheckpoint(latest)).rejects.toThrow(
      /checkpoint|complete|pending/i
    );
    expect(third.modelCalls).not.toHaveBeenCalled();
    expect(third.hooks).toEqual([]);
    expect(writes).not.toHaveBeenCalled();
    expect(fixture.effects).toEqual(['once']);
  });

  it('does not turn an actual HITL interrupt into approval; explicit native resume still works', async () => {
    const saver = new MemorySaver();
    const answers: string[] = [];
    const ask = tool(
      async ({ question }) => {
        const { answer } = askUserQuestion({ question });
        answers.push(answer);
        return answer;
      },
      {
        name: 'ask_user_question',
        description: 'Wait for an explicit answer.',
        schema: z.object({ question: z.string() }),
      }
    );
    const policy: NonNullable<t.RunConfig['checkpointPolicy']> = {
      threadId: THREAD,
      durability: 'sync',
    };
    const first = await createRun({
      saver,
      policy,
      tools: [ask],
      model: new FakeChatModel({
        responses: ['asking'],
        toolCalls: [
          {
            name: 'ask_user_question',
            args: { question: 'approve?' },
            id: 'pending-approval',
            type: 'tool_call',
          },
        ],
      }),
    });
    await first.run.processStream(input(), config());
    expect(first.run.getInterrupt()).toBeDefined();
    expect(answers).toEqual([]);
    const exact = await checkpoint(saver);
    const actual = await saver.getTuple(exact);
    expect(actual?.checkpoint.channel_values.runStepState).toMatchObject({
      runId: RUN,
    });
    const second = await createRun({ saver, policy, tools: [ask] });
    const writes = jest.spyOn(saver, 'put');
    await expect(second.run.continueFromCheckpoint(exact)).rejects.toThrow(
      /interrupt|approval/i
    );
    expect(second.modelCalls).not.toHaveBeenCalled();
    expect(second.hooks).toEqual([]);
    expect(writes).not.toHaveBeenCalled();
    expect(answers).toEqual([]);
    await second.run.resume({ answer: 'approved explicitly' }, exact);
    expect(answers).toEqual(['approved explicitly']);
    expect(second.modelCalls).toHaveBeenCalledTimes(1);
    expect(second.run.getInterrupt()).toBeUndefined();
  });

  it('snapshots the strict policy and preserves ordinary processStream with omitted durability', async () => {
    const saver = new MemorySaver();
    const policy = {
      threadId: THREAD,
      durability: 'sync',
    } satisfies NonNullable<t.RunConfig['checkpointPolicy']>;
    const fixture = await createRun({ saver, policy });
    policy.threadId = 'mutated-after-construction';
    const result = await fixture.run.processStream(input(), config());
    expect(fixture.modelCalls).toHaveBeenCalledTimes(1);
    expect(fixture.hooks).toEqual(['start', 'prompt']);
    expect(JSON.stringify(result)).toContain('continued answer');
    const checkpoints = [];
    for await (const tuple of saver.list(config())) checkpoints.push(tuple);
    expect(checkpoints.length).toBeGreaterThan(1);
    await expect(
      fixture.run.processStream(input(), {
        ...config(),
        configurable: { thread_id: policy.threadId },
      })
    ).rejects.toThrow(/thread/i);
    expect(fixture.modelCalls).toHaveBeenCalledTimes(1);
  });

  it.each(['process', 'resume', 'continue'] as const)(
    'rejects weaker durability and a different thread at the %s entry',
    async (entry) => {
      const saver = new MemorySaver();
      const fixture = await createRun({
        saver,
        policy: { threadId: THREAD, durability: 'sync' },
      });
      const writes = jest.spyOn(saver, 'put');
      const invoke = (caller: t.RunStreamConfig) => {
        if (entry === 'process')
          return fixture.run.processStream(input(), caller);
        if (entry === 'resume')
          return fixture.run.resume({ answer: 'not permission' }, caller);
        return fixture.run.continueFromCheckpoint(caller);
      };
      for (const durability of ['exit', 'async'] as const) {
        await expect(invoke({ ...config(), durability })).rejects.toThrow(
          /durability|sync/i
        );
      }
      await expect(
        invoke({ ...config(), configurable: { thread_id: 'different' } })
      ).rejects.toThrow(/thread/i);
      expect(fixture.modelCalls).not.toHaveBeenCalled();
      expect(fixture.hooks).toEqual([]);
      expect(writes).not.toHaveBeenCalled();
    }
  );

  it('cannot enable a strict checkpoint policy without an explicit saver', async () => {
    await expect(
      createRun({
        policy: { threadId: THREAD, durability: 'sync' },
        fallback: true,
      })
    ).rejects.toThrow(/checkpoint|saver|persist/i);
  });

  it('requires a fresh Run instead of reusing cleared or unrelated live sidecars', async () => {
    const fixture = await pending();
    const writes = jest.spyOn(fixture.saver, 'put');
    const calls = fixture.first.modelCalls.mock.calls.length;
    await expect(
      fixture.first.run.continueFromCheckpoint(fixture.exact)
    ).rejects.toThrow(/fresh|used|started/i);
    expect(writes).not.toHaveBeenCalled();
    expect(fixture.first.modelCalls).toHaveBeenCalledTimes(calls);
    expect(fixture.effects).toEqual(['once']);
  });

  it('admits only one racing continuation on a fresh Run after the actual checkpoint reads', async () => {
    const fixture = await pending();
    const second = await createRun({
      saver: fixture.saver,
      policy: fixture.policy,
      tools: [fixture.echo],
    });
    const results = await Promise.allSettled([
      second.run.continueFromCheckpoint(fixture.exact),
      second.run.continueFromCheckpoint(fixture.exact),
    ]);
    expect(results.filter(({ status }) => status === 'fulfilled')).toHaveLength(
      1
    );
    expect(results.filter(({ status }) => status === 'rejected')).toHaveLength(
      1
    );
    expect(second.modelCalls).toHaveBeenCalledTimes(1);
    expect(fixture.effects).toEqual(['once']);
  });

  it('does not replay a successful parallel task whose real pending writes survived a sibling failure', async () => {
    const saver = new MemorySaver();
    let successfulTaskSaved: (() => void) | undefined;
    let deadline: NodeJS.Timeout | undefined;
    const saved = new Promise<void>((resolve, reject) => {
      deadline = setTimeout(
        () => reject(new Error('Actual sibling save missing')),
        3000
      );
      successfulTaskSaved = () => {
        clearTimeout(deadline);
        resolve();
      };
    });
    const realPutWrites = saver.putWrites.bind(saver);
    const observedWrites = jest
      .spyOn(saver, 'putWrites')
      .mockImplementation(async (caller, writes, taskId) => {
        await realPutWrites(caller, writes, taskId);
        if ((caller.configurable?.checkpoint_ns ?? '') === '') {
          const tuple = await saver.getTuple(caller);
          if (
            tuple?.pendingWrites?.some(
              ([, channel, value]) =>
                channel === 'messages' &&
                Array.isArray(value) &&
                value.some(
                  (message) =>
                    isBaseMessage(message) && message.getType() === 'ai'
                )
            ) === true
          ) {
            successfulTaskSaved?.();
          }
        }
      });
    const firstModel = new SiblingModel(async (agent) => {
      if (agent === 'B') {
        await saved;
        throw new Error(
          'Synthetic provider unavailable after sibling persisted'
        );
      }
    });
    const first = await createSiblings(saver, firstModel);
    try {
      await expect(first.processStream(input(), config())).rejects.toThrow(
        'Synthetic provider unavailable after sibling persisted'
      );
    } finally {
      clearTimeout(deadline);
      successfulTaskSaved?.();
      observedWrites.mockRestore();
    }
    expect(firstModel.visits.sort()).toEqual(['A', 'B']);
    const exact = await checkpoint(saver);
    const tuple = await saver.getTuple(exact);
    expect(tuple?.pendingWrites).toEqual(
      expect.arrayContaining([expect.arrayContaining(['messages'])])
    );
    const secondModel = new SiblingModel(async () => {});
    const second = await createSiblings(saver, secondModel);
    await second.continueFromCheckpoint(exact);
    expect(secondModel.visits).toEqual(['B']);
    const latest = await saver.getTuple(config());
    const messages = latest?.checkpoint.channel_values.messages;
    if (!Array.isArray(messages))
      throw new Error('Expected actual checkpoint messages');
    expect(
      messages
        .filter(isBaseMessage)
        .filter((message) => message.getType() === 'human')
    ).toHaveLength(1);
  });
});

describe('prepared checkpoint input boundary', () => {
  it.each([
    new HumanMessage('native human'),
    new HumanMessageChunk('native human chunk'),
    new AIMessage('native AI'),
    new AIMessageChunk('native AI chunk'),
    new SystemMessage('native system'),
    new SystemMessageChunk('native system chunk'),
    new ChatMessage({ content: 'native chat', role: 'user' }),
    new ChatMessageChunk({ content: 'native chat chunk', role: 'user' }),
    new FunctionMessage({ content: 'native function', name: 'local' }),
    new FunctionMessageChunk({ content: 'native function chunk', name: 'local' }),
    new ToolMessage({ content: 'native tool', tool_call_id: 'local-tool' }),
    new ToolMessageChunk({ content: 'native tool chunk', tool_call_id: 'local-tool' }),
    new RemoveMessage({ id: 'source-human' }),
  ].map((message) => ({ nativeName: message.constructor.name, message })))('preserves native $nativeName through inspection and checkpointing', async ({ message }) => {
    const saver = new MemorySaver();
    const put = jest.spyOn(saver, 'put');
    const inspect = jest.fn(({ messages }: { messages: readonly BaseMessage[] }) => {
      const detached = messages[1];
      expect(Object.getPrototypeOf(detached)).toBe(Object.getPrototypeOf(message));
      expect(detached).not.toBe(message);
      expect(detached.content).toEqual(message.content);
      expect(Object.isFrozen(detached)).toBe(true);
      expect(detached.additional_kwargs).toEqual({ sourceMessageId: 'native-source' });
      expect(detached.getType()).toBe(message.getType());
    });
    const fixture = await createRun({
      saver,
      policy: { threadId: THREAD, durability: 'sync' },
      inspectCheckpointInput: inspect,
    });
    message.additional_kwargs = { sourceMessageId: 'native-source' };
    await fixture.run.processStream({ messages: [...input().messages, message, new HumanMessage('continue')] }, config());
    expect(inspect).toHaveBeenCalledTimes(1);
    expect(put).toHaveBeenCalled();
    expect(fixture.modelCalls).toHaveBeenCalledTimes(1);
  });

  // A frozen message must not retain a caller-controlled prototype whose inherited
  // content can change after inspection and before the first checkpoint.
  it.each(['inherited accessor', 'subclass'] as const)(
    'rejects a custom message %s before inspection, checkpoint or provider',
    async (kind) => {
      const saver = new MemorySaver();
      const put = jest.spyOn(saver, 'put');
      const writes = jest.spyOn(saver, 'putWrites');
      const inspect = jest.fn();
      const fixture = await createRun({
        saver,
        policy: { threadId: THREAD, durability: 'sync' },
        inspectCheckpointInput: inspect,
      });
      class CustomHumanMessage extends HumanMessage {}
      const original = input();
      if (kind === 'subclass') {
        original.messages[0] = new CustomHumanMessage('original prompt');
      } else {
        const custom = Object.create(HumanMessage.prototype);
        Object.defineProperties(custom, {
          content: { get() { return this.reviewText; }, configurable: true },
          reviewText: { value: 'original prompt', writable: true },
        });
        Object.setPrototypeOf(original.messages[0], custom);
        Reflect.deleteProperty(original.messages[0], 'content');
      }
      await expect(fixture.run.processStream(original, config())).rejects.toThrow(
        'Invalid prepared checkpoint input'
      );
      expect(inspect).not.toHaveBeenCalled();
      expect(put).not.toHaveBeenCalled();
      expect(writes).not.toHaveBeenCalled();
      expect(fixture.modelCalls).not.toHaveBeenCalled();
    }
  );

  it('inspects post-hook detached input before any checkpoint and preserves the inspected bytes', async () => {
    const saver = new MemorySaver();
    const put = jest.spyOn(saver, 'put');
    const writes = jest.spyOn(saver, 'putWrites');
    let entered!: () => void;
    let release!: () => void;
    const inspecting = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const barrier = new Promise<void>((resolve) => {
      release = resolve;
    });
    let inspected: readonly BaseMessage[] = [];
    const fixture = await createRun({
      saver,
      policy: { threadId: THREAD, durability: 'sync' },
      hookText: 'trusted hook context',
      inspectCheckpointInput: async ({ messages }) => {
        inspected = messages;
        expect(Object.isFrozen(messages)).toBe(true);
        expect(
          messages.some((message) => message.content === 'trusted hook context')
        ).toBe(true);
        entered();
        await barrier;
      },
    });
    const original = input();
    original.messages[0].additional_kwargs = { nested: { value: 'before' } };
    const pending = fixture.run.processStream(original, config());
    await Promise.race([
      inspecting,
      pending.then(() => {
        throw new Error('Inspection was not reached');
      }),
    ]);
    expect(put).not.toHaveBeenCalled();
    expect(writes).not.toHaveBeenCalled();
    expect(fixture.modelCalls).not.toHaveBeenCalled();
    original.messages[0].content = 'late caller mutation';
    (original.messages[0].additional_kwargs.nested as { value: string }).value =
      'after';
    release();
    await pending;
    const messages = (await saver.getTuple(config()))!.checkpoint.channel_values
      .messages as BaseMessage[];
    expect(messages[0].content).toBe('original prompt');
    expect(messages[0].additional_kwargs.nested).toEqual({ value: 'before' });
    expect(inspected[0].content).toBe('original prompt');
    expect(fixture.modelCalls).toHaveBeenCalledTimes(1);
    expect(put).toHaveBeenCalled();
  });

  it.each(['rejected', 'accessor', 'cycle'] as const)(
    'denies %s input before any durable write or model',
    async (reason) => {
      const saver = new MemorySaver();
      const put = jest.spyOn(saver, 'put');
      const writes = jest.spyOn(saver, 'putWrites');
      const accessor = jest.fn(() => 'unsafe');
      const fixture = await createRun({
        saver,
        policy: { threadId: THREAD, durability: 'sync' },
        hookText: 'unaccepted hook context',
        inspectCheckpointInput: ({ messages }) => {
          expect(
            messages.some(
              (message) => message.content === 'unaccepted hook context'
            )
          ).toBe(true);
          throw new Error('prepared input rejected');
        },
      });
      const original = input();
      if (reason === 'accessor')
        Object.defineProperty(
          original.messages[0].additional_kwargs,
          'unsafe',
          { get: accessor }
        );
      if (reason === 'cycle')
        original.messages[0].additional_kwargs.cycle = original.messages[0];
      await expect(
        fixture.run.processStream(original, config())
      ).rejects.toThrow();
      expect(accessor).not.toHaveBeenCalled();
      expect(put).not.toHaveBeenCalled();
      expect(writes).not.toHaveBeenCalled();
      expect(fixture.modelCalls).not.toHaveBeenCalled();
    }
  );
});
