/** Both real client paths must retain history without putting capability flags on the wire. */
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionCreateParamsNonStreaming,
} from 'openai/resources/chat/completions';
import { ChatOpenAI } from './index';

class ObservableChatOpenAI extends ChatOpenAI {
  get completionDelegate() {
    return this.completions;
  }
}

function reasoning(
  message: ChatCompletionMessageParam & { reasoning_content?: string }
): string | undefined {
  return message.reasoning_content;
}

describe.each([false, true])('complete history streaming=%s', (streaming) => {
  it.each([undefined, false, true])(
    'uses the real converter with opt-in=%s',
    async (capability) => {
      const client = new ObservableChatOpenAI({
        model: 'synthetic-k3',
        apiKey: 'synthetic-only',
        streaming,
        includeReasoningContent: true,
        preserveAssistantHistory: capability,
      });
      const observed: ChatCompletionCreateParams[] = [];
      async function completion(
        request: ChatCompletionCreateParamsStreaming
      ): Promise<AsyncIterable<ChatCompletionChunk>>;
      async function completion(
        request: ChatCompletionCreateParamsNonStreaming
      ): Promise<ChatCompletion>;
      async function completion(
        request: ChatCompletionCreateParams
      ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
        observed.push(request);
        if (request.stream === true) {
          return (async function* (): AsyncGenerator<ChatCompletionChunk> {
            yield {
              id: 'synthetic',
              object: 'chat.completion.chunk',
              created: 1,
              model: 'synthetic-k3',
              choices: [
                {
                  index: 0,
                  delta: { role: 'assistant', content: 'ok' },
                  finish_reason: null,
                  logprobs: null,
                },
              ],
            };
            yield {
              id: 'synthetic',
              object: 'chat.completion.chunk',
              created: 1,
              model: 'synthetic-k3',
              choices: [
                { index: 0, delta: {}, finish_reason: 'stop', logprobs: null },
              ],
            };
          })();
        }
        return {
          id: 'synthetic',
          object: 'chat.completion',
          created: 1,
          model: 'synthetic-k3',
          choices: [
            {
              index: 0,
              message: { role: 'assistant', content: 'ok', refusal: null },
              finish_reason: 'stop',
              logprobs: null,
            },
          ],
        };
      }
      client.completionDelegate.completionWithRetry = completion;
      await client.invoke([
        new AIMessage({
          content: '  ordinary  ',
          additional_kwargs: { reasoning_content: '普通 🧠' },
        }),
        new HumanMessage('next'),
        new AIMessage({
          content: 'checking',
          additional_kwargs: { reasoning_content: 'tool reason' },
          tool_calls: [{ id: 'c1', name: 'lookup', args: { q: 'ä' } }],
        }),
        new ToolMessage({ content: 'result', tool_call_id: 'c1' }),
      ]);
      expect(observed).toHaveLength(1);
      const request = observed[0];
      const [ordinary, , toolAssistant, toolResult] = request.messages;
      if (toolAssistant.role !== 'assistant' || toolResult.role !== 'tool') {
        throw new Error('Expected assistant/tool request sequence');
      }
      expect(ordinary.content).toBe('  ordinary  ');
      expect(reasoning(ordinary)).toBe(capability === true ? '普通 🧠' : undefined);
      expect(toolAssistant.content).toBe(capability === true ? 'checking' : '');
      expect(reasoning(toolAssistant)).toBe('tool reason');
      expect(toolResult.tool_call_id).toBe(toolAssistant.tool_calls?.[0].id);
      expect(request).not.toHaveProperty('preserveAssistantHistory');
      expect(request).not.toHaveProperty('includeReasoningContent');
    }
  );
});
