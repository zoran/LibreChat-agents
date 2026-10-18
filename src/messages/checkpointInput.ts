import { types } from 'node:util';
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
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';

// Native prototypes are library-owned. Arbitrary subclasses and inserted prototype
// layers retain caller-controlled data/accessors even after their instance is frozen.
// Do not use instanceof: LangChain's Symbol.hasInstance accepts structural impostors
// and reads their properties before our own-data validation.
const messagePrototypes = new Set<object>([
  AIMessage.prototype,
  AIMessageChunk.prototype,
  ChatMessage.prototype,
  ChatMessageChunk.prototype,
  FunctionMessage.prototype,
  FunctionMessageChunk.prototype,
  HumanMessage.prototype,
  HumanMessageChunk.prototype,
  RemoveMessage.prototype,
  SystemMessage.prototype,
  SystemMessageChunk.prototype,
  ToolMessage.prototype,
  ToolMessageChunk.prototype,
]);

const MAX_MESSAGES = 2048;
const MAX_NODES = 100_000;
const MAX_BYTES = 16 * 1024 * 1024;
const invalid = (): never => {
  throw new Error('Invalid prepared checkpoint input');
};

/** Detach own data while preserving native message prototypes; getters grant no input authority. */
function copyMessages(
  input: readonly BaseMessage[],
  frozen: boolean
): BaseMessage[] {
  let nodes = MAX_NODES;
  let bytes = MAX_BYTES;
  const ancestors = new Set<object>();
  const copies = new Map<object, object>();
  const copy = (value: unknown, depth: number): unknown => {
    if (--nodes < 0 || depth > 64) return invalid();
    if (typeof value === 'string') {
      bytes -= Buffer.byteLength(value, 'utf8');
      if (bytes < 0) return invalid();
      return value;
    }
    if (
      value == null ||
      typeof value === 'boolean' ||
      typeof value === 'undefined'
    )
      return value;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value !== 'object' || types.isProxy(value)) return invalid();
    if (ancestors.has(value)) return invalid();
    if (copies.has(value)) return copies.get(value);
    const prototype = Object.getPrototypeOf(value);
    const array = Array.isArray(value);
    const message = messagePrototypes.has(prototype);
    if (
      (array && prototype !== Array.prototype) ||
      (!array && !message && prototype !== Object.prototype && prototype !== null)
    )
      return invalid();
    const descriptors = Object.getOwnPropertyDescriptors(value);
    if (
      message &&
      (!Object.hasOwn(descriptors, 'content') ||
        Object.getOwnPropertyDescriptor(value, Symbol.for('langchain.message'))?.value !== true)
    )
      return invalid();
    const keys = Reflect.ownKeys(descriptors);
    if (keys.length > nodes) return invalid();
    const result = array ? [] : Object.create(prototype);
    copies.set(value, result);
    ancestors.add(value);
    for (const key of keys) {
      if (
        typeof key !== 'string' &&
        (key !== Symbol.for('langchain.message') || !message)
      )
        return invalid();
      const descriptor = (
        descriptors as Record<PropertyKey, PropertyDescriptor>
      )[key];
      if (!('value' in descriptor)) return invalid();
      if (array && key === 'length') {
        if (
          !Number.isSafeInteger(descriptor.value) ||
          descriptor.value > MAX_NODES
        )
          return invalid();
        continue;
      }
      bytes -= Buffer.byteLength(String(key), 'utf8');
      if (bytes < 0) return invalid();
      Object.defineProperty(result, key, {
        value: copy(descriptor.value, depth + 1),
        enumerable: descriptor.enumerable,
        configurable: true,
        writable: true,
      });
    }
    if (array) result.length = descriptors.length.value;
    ancestors.delete(value);
    return frozen ? Object.freeze(result) : result;
  };
  if (
    !Array.isArray(input) ||
    types.isProxy(input) ||
    input.length > MAX_MESSAGES
  )
    return invalid();
  const result = copy(input, 0) as BaseMessage[];
  for (let index = 0; index < result.length; index++) {
    const message: unknown = result[index];
    if (
      typeof message !== 'object' || message === null ||
      !messagePrototypes.has(Object.getPrototypeOf(message))
    )
      return invalid();
  }
  return result;
}

/** The inspector and graph receive equivalent, disjoint copies of the exact post-hook input. */
export function snapshotCheckpointInput(messages: readonly BaseMessage[]): {
  inspectionMessages: readonly BaseMessage[];
  messages: BaseMessage[];
} {
  const inspectionMessages = copyMessages(messages, true);
  return {
    inspectionMessages,
    messages: copyMessages(inspectionMessages, false),
  };
}
