/** Complete-history regression cases use the real formatter and OpenAI converter. */
import { AIMessage } from '@langchain/core/messages';
import type { TPayload, MessageContentComplex } from '@/types';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { getProviderMessageProvenance } from './provenance';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

type WireMessage = ReturnType<typeof _convertMessagesToOpenAIParams>[number] & {
  reasoning_content?: string;
};

function reasoning(message: WireMessage): string | undefined {
  return message.reasoning_content;
}

function assistant(
  message: WireMessage
): Extract<WireMessage, { role: 'assistant' }> {
  if (message.role !== 'assistant') throw new Error('Expected assistant message');
  return message;
}

function tool(message: WireMessage): Extract<WireMessage, { role: 'tool' }> {
  if (message.role !== 'tool') throw new Error('Expected tool message');
  return message;
}

const text = (value: string, ids?: string[]): MessageContentComplex => ({
  type: ContentTypes.TEXT,
  text: value,
  ...(ids ? { tool_call_ids: ids } : {}),
});
const think = (value: string): MessageContentComplex => ({
  type: ContentTypes.THINK,
  think: value,
});
const call = (id: string, name = 'lookup'): MessageContentComplex => ({
  type: ContentTypes.TOOL_CALL,
  tool_call: { id, name, args: '{"q":"ä"}', output: 'synthetic result' },
});
function replay(
  content: MessageContentComplex[],
  preserveAssistantHistory?: boolean,
  tools = new Set(['lookup'])
) {
  const payload = [{
    role: 'assistant',
    messageId: 'synthetic-1',
    isCreatedByUser: false,
    content,
  }] as TPayload;
  const result = formatAgentMessages(payload, { 0: 100 }, tools, undefined, {
    preserveAssistantHistory,
    preserveReasoningContent: true,
  });
  return {
    ...result,
    wire: _convertMessagesToOpenAIParams(result.messages, 'synthetic-k3', {
      preserveAssistantHistory,
      includeReasoningContent: true,
    }),
  };
}

describe('explicit complete assistant history', () => {
  it.each(['Überlegung 中文 🧠', '', '  \n'])(
    'preserves ordinary reasoning %j and exact text',
    (reasoningText) => {
      const { wire } = replay(
        [think(reasoningText), text('  Antwort\n'), text('二  ')],
        true
      );
      expect(wire).toHaveLength(1);
      expect(wire[0].content).toBe('  Antwort\n二  ');
      expect(reasoning(wire[0]) ?? '').toBe(reasoningText);
    }
  );
  it('retains a reasoning-only turn without a visible filler answer', () => {
    const { wire, messages } = replay([think('Nur Denken 🧠'), text('')], true);
    expect(wire).toHaveLength(1);
    expect(wire[0].content).toBe('');
    expect(reasoning(wire[0])).toBe('Nur Denken 🧠');
    expect(getProviderMessageProvenance(messages[0])?.parts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sourceMessageId: 'synthetic-1',
          sourceContentPartIndices: expect.arrayContaining([0]),
        }),
      ])
    );
  });
  it.each([false, true])('binds mixed text and parallel calls with marker=%s', (marked) => {
    const { wire } = replay([
      think('first 🧠'), text('  Checking\n', marked ? ['c1', 'c2'] : undefined),
      call('c1'), call('c2'), think('second'), text(' Final  '),
    ], true);
    expect(wire.map((m) => m.role)).toEqual(['assistant', 'tool', 'tool', 'assistant']);
    expect(wire[0].content).toBe('  Checking\n');
    expect(assistant(wire[0]).tool_calls?.map((call) => call.id)).toEqual(['c1', 'c2']);
    expect(reasoning(wire[0])).toBe('first 🧠');
    expect(tool(wire[1]).tool_call_id).toBe('c1');
    expect(tool(wire[2]).tool_call_id).toBe('c2');
    expect(wire[3].content).toBe(' Final  ');
    expect(reasoning(wire[3])).toBe('second');
  });
  it('starts a separate assistant anchor for each sequential tool cycle', () => {
    const { wire } = replay(
      [think('r1'), text('t1'), call('c1'), think('r2'), text('t2'), call('c2')],
      true
    );
    expect(wire.map((m) => m.role)).toEqual(['assistant', 'tool', 'assistant', 'tool']);
    expect(wire[0].content).toBe('t1');
    expect(wire[2].content).toBe('t2');
    expect(assistant(wire[2]).tool_calls?.[0].id).toBe('c2');
    expect(reasoning(wire[2])).toBe('r2');
  });
  it('retains typed mixed media rather than flattening it', () => {
    const image = {
      type: 'image_url',
      image_url: { url: 'data:image/png;base64,AA==' },
    } as MessageContentComplex;
    const { wire } = replay([think('r'), text('caption'), image, call('c1')], true);
    expect(Array.isArray(wire[0].content)).toBe(true);
    expect(wire[0].content).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'image_url' })])
    );
  });
  it('never restores filtered tool calls or results', () => {
    const { wire } = replay([call('removed', 'not_authorized')], true);
    expect(wire).toEqual([]);
    const allowed = replay([text('own text'), call('removed', 'not_authorized')], true);
    expect(JSON.stringify(allowed.wire)).not.toContain('synthetic result');
    expect(JSON.stringify(allowed.wire)).not.toContain('not_authorized');
  });
  it('preserves absent/false legacy behavior and does not synthesize reasoning', () => {
    const parts = [think('r'), text('answer')];
    expect(replay(parts).wire).toEqual(replay(parts, false).wire);
    expect(reasoning(replay(parts).wire[0])).toBeUndefined();
    const wire = _convertMessagesToOpenAIParams(
      [new AIMessage('plain')],
      'synthetic-k3',
      { preserveAssistantHistory: true }
    );
    expect(reasoning(wire[0])).toBeUndefined();
  });
});
