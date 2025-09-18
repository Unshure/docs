# `strands.event_loop`

This package provides the core event loop implementation for the agents SDK.

The event loop enables conversational AI agents to process messages, execute tools, and handle errors in a controlled, iterative manner.

## `strands.event_loop.event_loop`

This module implements the central event loop.

The event loop allows agents to:

1. Process conversation messages
1. Execute tools based on model requests
1. Handle errors and recovery strategies
1. Manage recursive execution cycles

### `event_loop_cycle(agent, invocation_state)`

Execute a single cycle of the event loop.

This core function processes a single conversation turn, handling model inference, tool execution, and error recovery. It manages the entire lifecycle of a conversation turn, including:

1. Initializing cycle state and metrics
1. Checking execution limits
1. Processing messages with the model
1. Handling tool execution requests
1. Managing recursive calls for multi-turn tool interactions
1. Collecting and reporting metrics
1. Error handling and recovery

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `Agent` | The agent for which the cycle is being executed. | *required* | | `invocation_state` | `dict[str, Any]` | Additional arguments including: request_state: State maintained across cycles event_loop_cycle_id: Unique ID for this cycle event_loop_cycle_span: Current tracing Span for this cycle | *required* |

Yields:

| Type | Description | | --- | --- | | `AsyncGenerator[TypedEvent, None]` | Model and tool stream events. The last event is a tuple containing: StopReason: Reason the model stopped generating (e.g., "tool_use") Message: The generated message from the model EventLoopMetrics: Updated metrics for the event loop Any: Updated request state |

Raises:

| Type | Description | | --- | --- | | `EventLoopException` | If an error occurs during execution | | `ContextWindowOverflowException` | If the input is too large for the model |

Source code in `strands/event_loop/event_loop.py`

```
async def event_loop_cycle(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
    """Execute a single cycle of the event loop.

    This core function processes a single conversation turn, handling model inference, tool execution, and error
    recovery. It manages the entire lifecycle of a conversation turn, including:

    1. Initializing cycle state and metrics
    2. Checking execution limits
    3. Processing messages with the model
    4. Handling tool execution requests
    5. Managing recursive calls for multi-turn tool interactions
    6. Collecting and reporting metrics
    7. Error handling and recovery

    Args:
        agent: The agent for which the cycle is being executed.
        invocation_state: Additional arguments including:

            - request_state: State maintained across cycles
            - event_loop_cycle_id: Unique ID for this cycle
            - event_loop_cycle_span: Current tracing Span for this cycle

    Yields:
        Model and tool stream events. The last event is a tuple containing:

            - StopReason: Reason the model stopped generating (e.g., "tool_use")
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state

    Raises:
        EventLoopException: If an error occurs during execution
        ContextWindowOverflowException: If the input is too large for the model
    """
    # Initialize cycle state
    invocation_state["event_loop_cycle_id"] = uuid.uuid4()

    # Initialize state and get cycle trace
    if "request_state" not in invocation_state:
        invocation_state["request_state"] = {}
    attributes = {"event_loop_cycle_id": str(invocation_state.get("event_loop_cycle_id"))}
    cycle_start_time, cycle_trace = agent.event_loop_metrics.start_cycle(attributes=attributes)
    invocation_state["event_loop_cycle_trace"] = cycle_trace

    yield StartEvent()
    yield StartEventLoopEvent()

    # Create tracer span for this event loop cycle
    tracer = get_tracer()
    cycle_span = tracer.start_event_loop_cycle_span(
        invocation_state=invocation_state, messages=agent.messages, parent_span=agent.trace_span
    )
    invocation_state["event_loop_cycle_span"] = cycle_span

    # Create a trace for the stream_messages call
    stream_trace = Trace("stream_messages", parent_id=cycle_trace.id)
    cycle_trace.add_child(stream_trace)

    # Process messages with exponential backoff for throttling
    message: Message
    stop_reason: StopReason
    usage: Any
    metrics: Metrics

    # Retry loop for handling throttling exceptions
    current_delay = INITIAL_DELAY
    for attempt in range(MAX_ATTEMPTS):
        model_id = agent.model.config.get("model_id") if hasattr(agent.model, "config") else None
        model_invoke_span = tracer.start_model_invoke_span(
            messages=agent.messages,
            parent_span=cycle_span,
            model_id=model_id,
        )
        with trace_api.use_span(model_invoke_span):
            agent.hooks.invoke_callbacks(
                BeforeModelInvocationEvent(
                    agent=agent,
                )
            )

            tool_specs = agent.tool_registry.get_all_tool_specs()

            try:
                async for event in stream_messages(agent.model, agent.system_prompt, agent.messages, tool_specs):
                    if not isinstance(event, ModelStopReason):
                        yield event

                stop_reason, message, usage, metrics = event["stop"]
                invocation_state.setdefault("request_state", {})

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        stop_response=AfterModelInvocationEvent.ModelStopResponse(
                            stop_reason=stop_reason,
                            message=message,
                        ),
                    )
                )

                if stop_reason == "max_tokens":
                    message = recover_message_on_max_tokens_reached(message)

                if model_invoke_span:
                    tracer.end_model_invoke_span(model_invoke_span, message, usage, stop_reason)
                break  # Success! Break out of retry loop

            except Exception as e:
                if model_invoke_span:
                    tracer.end_span_with_error(model_invoke_span, str(e), e)

                agent.hooks.invoke_callbacks(
                    AfterModelInvocationEvent(
                        agent=agent,
                        exception=e,
                    )
                )

                if isinstance(e, ModelThrottledException):
                    if attempt + 1 == MAX_ATTEMPTS:
                        yield ForceStopEvent(reason=e)
                        raise e

                    logger.debug(
                        "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
                        "| throttling exception encountered "
                        "| delaying before next retry",
                        current_delay,
                        MAX_ATTEMPTS,
                        attempt + 1,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * 2, MAX_DELAY)

                    yield EventLoopThrottleEvent(delay=current_delay)
                else:
                    raise e

    try:
        # Add message in trace and mark the end of the stream messages trace
        stream_trace.add_message(message)
        stream_trace.end()

        # Add the response message to the conversation
        agent.messages.append(message)
        agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=message))
        yield ModelMessageEvent(message=message)

        # Update metrics
        agent.event_loop_metrics.update_usage(usage)
        agent.event_loop_metrics.update_metrics(metrics)

        if stop_reason == "max_tokens":
            """
            Handle max_tokens limit reached by the model.

            When the model reaches its maximum token limit, this represents a potentially unrecoverable
            state where the model's response was truncated. By default, Strands fails hard with an
            MaxTokensReachedException to maintain consistency with other failure types.
            """
            raise MaxTokensReachedException(
                message=(
                    "Agent has reached an unrecoverable state due to max_tokens limit. "
                    "For more information see: "
                    "https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/#maxtokensreachedexception"
                )
            )

        # If the model is requesting to use tools
        if stop_reason == "tool_use":
            # Handle tool execution
            events = _handle_tool_execution(
                stop_reason,
                message,
                agent=agent,
                cycle_trace=cycle_trace,
                cycle_span=cycle_span,
                cycle_start_time=cycle_start_time,
                invocation_state=invocation_state,
            )
            async for typed_event in events:
                yield typed_event

            return

        # End the cycle and return results
        agent.event_loop_metrics.end_cycle(cycle_start_time, cycle_trace, attributes)
        if cycle_span:
            tracer.end_event_loop_cycle_span(
                span=cycle_span,
                message=message,
            )
    except EventLoopException as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Don't yield or log the exception - we already did it when we
        # raised the exception and we don't need that duplication.
        raise
    except (ContextWindowOverflowException, MaxTokensReachedException) as e:
        # Special cased exceptions which we want to bubble up rather than get wrapped in an EventLoopException
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)
        raise e
    except Exception as e:
        if cycle_span:
            tracer.end_span_with_error(cycle_span, str(e), e)

        # Handle any other exceptions
        yield ForceStopEvent(reason=e)
        logger.exception("cycle failed")
        raise EventLoopException(e, invocation_state["request_state"]) from e

    yield EventLoopStopEvent(stop_reason, message, agent.event_loop_metrics, invocation_state["request_state"])
```

### `recurse_event_loop(agent, invocation_state)`

Make a recursive call to event_loop_cycle with the current state.

This function is used when the event loop needs to continue processing after tool execution.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `Agent` | Agent for which the recursive call is being made. | *required* | | `invocation_state` | `dict[str, Any]` | Arguments to pass through event_loop_cycle | *required* |

Yields:

| Type | Description | | --- | --- | | `AsyncGenerator[TypedEvent, None]` | Results from event_loop_cycle where the last result contains: StopReason: Reason the model stopped generating Message: The generated message from the model EventLoopMetrics: Updated metrics for the event loop Any: Updated request state |

Source code in `strands/event_loop/event_loop.py`

```
async def recurse_event_loop(agent: "Agent", invocation_state: dict[str, Any]) -> AsyncGenerator[TypedEvent, None]:
    """Make a recursive call to event_loop_cycle with the current state.

    This function is used when the event loop needs to continue processing after tool execution.

    Args:
        agent: Agent for which the recursive call is being made.
        invocation_state: Arguments to pass through event_loop_cycle


    Yields:
        Results from event_loop_cycle where the last result contains:

            - StopReason: Reason the model stopped generating
            - Message: The generated message from the model
            - EventLoopMetrics: Updated metrics for the event loop
            - Any: Updated request state
    """
    cycle_trace = invocation_state["event_loop_cycle_trace"]

    # Recursive call trace
    recursive_trace = Trace("Recursive call", parent_id=cycle_trace.id)
    cycle_trace.add_child(recursive_trace)

    yield StartEvent()

    events = event_loop_cycle(agent=agent, invocation_state=invocation_state)
    async for event in events:
        yield event

    recursive_trace.end()
```

## `strands.event_loop.streaming`

Utilities for handling streaming responses from language models.

### `extract_usage_metrics(event)`

Extracts usage metrics from the metadata chunk.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `MetadataEvent` | metadata. | *required* |

Returns:

| Type | Description | | --- | --- | | `tuple[Usage, Metrics]` | The extracted usage metrics and latency. |

Source code in `strands/event_loop/streaming.py`

```
def extract_usage_metrics(event: MetadataEvent) -> tuple[Usage, Metrics]:
    """Extracts usage metrics from the metadata chunk.

    Args:
        event: metadata.

    Returns:
        The extracted usage metrics and latency.
    """
    usage = Usage(**event["usage"])
    metrics = Metrics(**event["metrics"])

    return usage, metrics
```

### `handle_content_block_delta(event, state)`

Handles content block delta updates by appending text, tool input, or reasoning content to the state.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `ContentBlockDeltaEvent` | Delta event. | *required* | | `state` | `dict[str, Any]` | The current state of message processing. | *required* |

Returns:

| Type | Description | | --- | --- | | `tuple[dict[str, Any], ModelStreamEvent]` | Updated state with appended text or tool input. |

Source code in `strands/event_loop/streaming.py`

```
def handle_content_block_delta(
    event: ContentBlockDeltaEvent, state: dict[str, Any]
) -> tuple[dict[str, Any], ModelStreamEvent]:
    """Handles content block delta updates by appending text, tool input, or reasoning content to the state.

    Args:
        event: Delta event.
        state: The current state of message processing.

    Returns:
        Updated state with appended text or tool input.
    """
    delta_content = event["delta"]

    typed_event: ModelStreamEvent = ModelStreamEvent({})

    if "toolUse" in delta_content:
        if "input" not in state["current_tool_use"]:
            state["current_tool_use"]["input"] = ""

        state["current_tool_use"]["input"] += delta_content["toolUse"]["input"]
        typed_event = ToolUseStreamEvent(delta_content, state["current_tool_use"])

    elif "text" in delta_content:
        state["text"] += delta_content["text"]
        typed_event = TextStreamEvent(text=delta_content["text"], delta=delta_content)

    elif "citation" in delta_content:
        if "citationsContent" not in state:
            state["citationsContent"] = []

        state["citationsContent"].append(delta_content["citation"])
        typed_event = CitationStreamEvent(delta=delta_content, citation=delta_content["citation"])

    elif "reasoningContent" in delta_content:
        if "text" in delta_content["reasoningContent"]:
            if "reasoningText" not in state:
                state["reasoningText"] = ""

            state["reasoningText"] += delta_content["reasoningContent"]["text"]
            typed_event = ReasoningTextStreamEvent(
                reasoning_text=delta_content["reasoningContent"]["text"],
                delta=delta_content,
            )

        elif "signature" in delta_content["reasoningContent"]:
            if "signature" not in state:
                state["signature"] = ""

            state["signature"] += delta_content["reasoningContent"]["signature"]
            typed_event = ReasoningSignatureStreamEvent(
                reasoning_signature=delta_content["reasoningContent"]["signature"],
                delta=delta_content,
            )

        elif redacted_content := delta_content["reasoningContent"].get("redactedContent"):
            state["redactedContent"] = state.get("redactedContent", b"") + redacted_content
            typed_event = ReasoningRedactedContentStreamEvent(redacted_content=redacted_content, delta=delta_content)

    return state, typed_event
```

### `handle_content_block_start(event)`

Handles the start of a content block by extracting tool usage information if any.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `ContentBlockStartEvent` | Start event. | *required* |

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | Dictionary with tool use id and name if tool use request, empty dictionary otherwise. |

Source code in `strands/event_loop/streaming.py`

```
def handle_content_block_start(event: ContentBlockStartEvent) -> dict[str, Any]:
    """Handles the start of a content block by extracting tool usage information if any.

    Args:
        event: Start event.

    Returns:
        Dictionary with tool use id and name if tool use request, empty dictionary otherwise.
    """
    start: ContentBlockStart = event["start"]
    current_tool_use = {}

    if "toolUse" in start and start["toolUse"]:
        tool_use_data = start["toolUse"]
        current_tool_use["toolUseId"] = tool_use_data["toolUseId"]
        current_tool_use["name"] = tool_use_data["name"]
        current_tool_use["input"] = ""

    return current_tool_use
```

### `handle_content_block_stop(state)`

Handles the end of a content block by finalizing tool usage, text content, or reasoning content.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state` | `dict[str, Any]` | The current state of message processing. | *required* |

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | Updated state with finalized content block. |

Source code in `strands/event_loop/streaming.py`

```
def handle_content_block_stop(state: dict[str, Any]) -> dict[str, Any]:
    """Handles the end of a content block by finalizing tool usage, text content, or reasoning content.

    Args:
        state: The current state of message processing.

    Returns:
        Updated state with finalized content block.
    """
    content: list[ContentBlock] = state["content"]

    current_tool_use = state["current_tool_use"]
    text = state["text"]
    reasoning_text = state["reasoningText"]
    citations_content = state["citationsContent"]
    redacted_content = state.get("redactedContent")

    if current_tool_use:
        if "input" not in current_tool_use:
            current_tool_use["input"] = ""

        try:
            current_tool_use["input"] = json.loads(current_tool_use["input"])
        except ValueError:
            current_tool_use["input"] = {}

        tool_use_id = current_tool_use["toolUseId"]
        tool_use_name = current_tool_use["name"]

        tool_use = ToolUse(
            toolUseId=tool_use_id,
            name=tool_use_name,
            input=current_tool_use["input"],
        )
        content.append({"toolUse": tool_use})
        state["current_tool_use"] = {}

    elif text:
        content.append({"text": text})
        state["text"] = ""
        if citations_content:
            citations_block: CitationsContentBlock = {"citations": citations_content}
            content.append({"citationsContent": citations_block})
            state["citationsContent"] = []

    elif reasoning_text:
        content_block: ContentBlock = {
            "reasoningContent": {
                "reasoningText": {
                    "text": state["reasoningText"],
                }
            }
        }

        if "signature" in state:
            content_block["reasoningContent"]["reasoningText"]["signature"] = state["signature"]

        content.append(content_block)
        state["reasoningText"] = ""
    elif redacted_content:
        content.append({"reasoningContent": {"redactedContent": redacted_content}})
        state["redactedContent"] = b""

    return state
```

### `handle_message_start(event, message)`

Handles the start of a message by setting the role in the message dictionary.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `MessageStartEvent` | A message start event. | *required* | | `message` | `Message` | The message dictionary being constructed. | *required* |

Returns:

| Type | Description | | --- | --- | | `Message` | Updated message dictionary with the role set. |

Source code in `strands/event_loop/streaming.py`

```
def handle_message_start(event: MessageStartEvent, message: Message) -> Message:
    """Handles the start of a message by setting the role in the message dictionary.

    Args:
        event: A message start event.
        message: The message dictionary being constructed.

    Returns:
        Updated message dictionary with the role set.
    """
    message["role"] = event["role"]
    return message
```

### `handle_message_stop(event)`

Handles the end of a message by returning the stop reason.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `MessageStopEvent` | Stop event. | *required* |

Returns:

| Type | Description | | --- | --- | | `StopReason` | The reason for stopping the stream. |

Source code in `strands/event_loop/streaming.py`

```
def handle_message_stop(event: MessageStopEvent) -> StopReason:
    """Handles the end of a message by returning the stop reason.

    Args:
        event: Stop event.

    Returns:
        The reason for stopping the stream.
    """
    return event["stopReason"]
```

### `handle_redact_content(event, state)`

Handles redacting content from the input or output.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `RedactContentEvent` | Redact Content Event. | *required* | | `state` | `dict[str, Any]` | The current state of message processing. | *required* |

Source code in `strands/event_loop/streaming.py`

```
def handle_redact_content(event: RedactContentEvent, state: dict[str, Any]) -> None:
    """Handles redacting content from the input or output.

    Args:
        event: Redact Content Event.
        state: The current state of message processing.
    """
    if event.get("redactAssistantContentMessage") is not None:
        state["message"]["content"] = [{"text": event["redactAssistantContentMessage"]}]
```

### `process_stream(chunks)`

Processes the response stream from the API, constructing the final message and extracting usage metrics.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `chunks` | `AsyncIterable[StreamEvent]` | The chunks of the response stream from the model. | *required* |

Yields:

| Type | Description | | --- | --- | | `AsyncGenerator[TypedEvent, None]` | The reason for stopping, the constructed message, and the usage metrics. |

Source code in `strands/event_loop/streaming.py`

```
async def process_stream(chunks: AsyncIterable[StreamEvent]) -> AsyncGenerator[TypedEvent, None]:
    """Processes the response stream from the API, constructing the final message and extracting usage metrics.

    Args:
        chunks: The chunks of the response stream from the model.

    Yields:
        The reason for stopping, the constructed message, and the usage metrics.
    """
    stop_reason: StopReason = "end_turn"

    state: dict[str, Any] = {
        "message": {"role": "assistant", "content": []},
        "text": "",
        "current_tool_use": {},
        "reasoningText": "",
        "citationsContent": [],
    }
    state["content"] = state["message"]["content"]

    usage: Usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
    metrics: Metrics = Metrics(latencyMs=0)

    async for chunk in chunks:
        yield ModelStreamChunkEvent(chunk=chunk)
        if "messageStart" in chunk:
            state["message"] = handle_message_start(chunk["messageStart"], state["message"])
        elif "contentBlockStart" in chunk:
            state["current_tool_use"] = handle_content_block_start(chunk["contentBlockStart"])
        elif "contentBlockDelta" in chunk:
            state, typed_event = handle_content_block_delta(chunk["contentBlockDelta"], state)
            yield typed_event
        elif "contentBlockStop" in chunk:
            state = handle_content_block_stop(state)
        elif "messageStop" in chunk:
            stop_reason = handle_message_stop(chunk["messageStop"])
        elif "metadata" in chunk:
            usage, metrics = extract_usage_metrics(chunk["metadata"])
        elif "redactContent" in chunk:
            handle_redact_content(chunk["redactContent"], state)

    yield ModelStopReason(stop_reason=stop_reason, message=state["message"], usage=usage, metrics=metrics)
```

### `remove_blank_messages_content_text(messages)`

Remove or replace blank text in message content.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `messages` | `Messages` | Conversation messages to update. | *required* |

Returns:

| Type | Description | | --- | --- | | `Messages` | Updated messages. |

Source code in `strands/event_loop/streaming.py`

```
def remove_blank_messages_content_text(messages: Messages) -> Messages:
    """Remove or replace blank text in message content.

    Args:
        messages: Conversation messages to update.

    Returns:
        Updated messages.
    """
    removed_blank_message_content_text = False
    replaced_blank_message_content_text = False

    for message in messages:
        # only modify assistant messages
        if "role" in message and message["role"] != "assistant":
            continue
        if "content" in message:
            content = message["content"]
            has_tool_use = any("toolUse" in item for item in content)
            if len(content) == 0:
                content.append({"text": "[blank text]"})
                continue

            if has_tool_use:
                # Remove blank 'text' items for assistant messages
                before_len = len(content)
                content[:] = [item for item in content if "text" not in item or item["text"].strip()]
                if not removed_blank_message_content_text and before_len != len(content):
                    removed_blank_message_content_text = True
            else:
                # Replace blank 'text' with '[blank text]' for assistant messages
                for item in content:
                    if "text" in item and not item["text"].strip():
                        replaced_blank_message_content_text = True
                        item["text"] = "[blank text]"

    if removed_blank_message_content_text:
        logger.debug("removed blank message context text")
    if replaced_blank_message_content_text:
        logger.debug("replaced blank message context text")

    return messages
```

### `stream_messages(model, system_prompt, messages, tool_specs)`

Streams messages to the model and processes the response.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model` | Model provider. | *required* | | `system_prompt` | `Optional[str]` | The system prompt to send. | *required* | | `messages` | `Messages` | List of messages to send. | *required* | | `tool_specs` | `list[ToolSpec]` | The list of tool specs. | *required* |

Yields:

| Type | Description | | --- | --- | | `AsyncGenerator[TypedEvent, None]` | The reason for stopping, the final message, and the usage metrics |

Source code in `strands/event_loop/streaming.py`

```
async def stream_messages(
    model: Model,
    system_prompt: Optional[str],
    messages: Messages,
    tool_specs: list[ToolSpec],
) -> AsyncGenerator[TypedEvent, None]:
    """Streams messages to the model and processes the response.

    Args:
        model: Model provider.
        system_prompt: The system prompt to send.
        messages: List of messages to send.
        tool_specs: The list of tool specs.

    Yields:
        The reason for stopping, the final message, and the usage metrics
    """
    logger.debug("model=<%s> | streaming messages", model)

    messages = remove_blank_messages_content_text(messages)
    chunks = model.stream(messages, tool_specs if tool_specs else None, system_prompt)

    async for event in process_stream(chunks):
        yield event
```
