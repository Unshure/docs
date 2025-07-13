# State & Sessions

## State Management

Strands Agents state is maintained in several forms:

1. **Conversation History:** The sequence of messages between the user and the agent.
1. **Agent State**: Stateful information outside of conversation context, maintained across multiple requests.
1. **Request State**: Contextual information maintained within a single request.

Understanding how state works in Strands is essential for building agents that can maintain context across multi-turn interactions and workflows.

### Conversation History

Conversation history is the primary form of context in a Strands agent, directly accessible through the `agent.messages` property:

```
from strands import Agent

# Create an agent
agent = Agent()

# Send a message and get a response
agent("Hello!")

# Access the conversation history
print(agent.messages)  # Shows all messages exchanged so far

```

The `agent.messages` list contains all user and assistant messages, including tool calls and tool results. This is the primary way to inspect what's happening in your agent's conversation.

You can initialize an agent with existing messages to continue a conversation or pre-fill your Agent's context with information:

```
from strands import Agent

# Create an agent with initial messages
agent = Agent(messages=[
    {"role": "user", "content": [{"text": "Hello, my name is Strands!"}]},
    {"role": "assistant", "content": [{"text": "Hi there! How can I help you today?"}]}
])

# Continue the conversation
agent("What's my name?")

```

Conversation history is automatically:

- Maintained between calls to the agent
- Passed to the model during each inference
- Used for tool execution context
- Managed to prevent context window overflow

#### Direct Tool Calling

Direct tool calls are (by default) recorded in the conversation history:

```
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])

# Direct tool call with recording (default behavior)
agent.tool.calculator(expression="123 * 456")

# Direct tool call without recording
agent.tool.calculator(expression="765 / 987", record_direct_tool_call=False)

print(agent.messages)

```

In this example we can see that the first `agent.tool.calculator()` call is recorded in the agent's conversation history.

The second `agent.tool.calculator()` call is **not** recorded in the history because we specified the `record_direct_tool_call=False` argument.

#### Conversation Manager

Strands uses a conversation manager to handle conversation history effectively. The default is the [`SlidingWindowConversationManager`](../../../../api-reference/agent/#strands.agent.conversation_manager.sliding_window_conversation_manager.SlidingWindowConversationManager), which keeps recent messages and removes older ones when needed:

```
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Create a conversation manager with custom window size
# By default, SlidingWindowConversationManager is used even if not specified
conversation_manager = SlidingWindowConversationManager(
    window_size=10,  # Maximum number of message pairs to keep
)

# Use the conversation manager with your agent
agent = Agent(conversation_manager=conversation_manager)

```

The sliding window conversation manager:

- Keeps the most recent N message pairs
- Removes the oldest messages when the window size is exceeded
- Handles context window overflow exceptions by reducing context
- Ensures conversations don't exceed model context limits

See [`Context Management`](../context-management/) for more information about conversation managers.

### Agent State

Agent state provides key-value storage for stateful information that exists outside of the conversation context. Unlike conversation history, agent state is not passed to the model during inference but can be accessed and modified by tools and application logic.

#### Basic Usage

```
from strands import Agent

# Create an agent with initial state
agent = Agent(state={"user_preferences": {"theme": "dark"}, "session_count": 0})


# Access state values
theme = agent.state.get("user_preferences")
print(theme)  # {"theme": "dark"}

# Set new state values
agent.state.set("last_action", "login")
agent.state.set("session_count", 1)

# Get entire state
all_state = agent.state.get()
print(all_state)  # All state data as a dictionary

# Delete state values
agent.state.delete("last_action")

```

#### State Validation and Safety

Agent state enforces JSON serialization validation to ensure data can be persisted and restored:

```
from strands import Agent

agent = Agent()

# Valid JSON-serializable values
agent.state.set("string_value", "hello")
agent.state.set("number_value", 42)
agent.state.set("boolean_value", True)
agent.state.set("list_value", [1, 2, 3])
agent.state.set("dict_value", {"nested": "data"})
agent.state.set("null_value", None)

# Invalid values will raise ValueError
try:
    agent.state.set("function", lambda x: x)  # Not JSON serializable
except ValueError as e:
    print(f"Error: {e}")

```

#### Using State in Tools

Agent state is particularly useful for maintaining information across tool executions:

```
from strands import Agent
from strands.tools.decorator import tool

@tool
def track_user_action(action: str, agent: Agent):
    """Track user actions in agent state."""
    # Get current action count
    action_count = agent.state.get("action_count") or 0

    # Update state
    agent.state.set("action_count", action_count + 1)
    agent.state.set("last_action", action)

    return f"Action '{action}' recorded. Total actions: {action_count + 1}"

@tool
def get_user_stats(agent: Agent):
    """Get user statistics from agent state."""
    action_count = agent.state.get("action_count") or 0
    last_action = agent.state.get("last_action") or "none"

    return f"Actions performed: {action_count}, Last action: {last_action}"

# Create agent with tools
agent = Agent(tools=[track_user_action, get_user_stats])

# Use tools that modify and read state
agent("Track that I logged in")
agent("Track that I viewed my profile")
print(f"Actions taken: {agent.state.get('action_count')}")
print(f"Last action: {agent.state.get('last_action')}")

```

### Request State

Each agent interaction maintains a request state dictionary that persists throughout the event loop cycles and is **not** included in the agent's context:

```
from strands import Agent

def custom_callback_handler(**kwargs):
    # Access request state
    if "request_state" in kwargs:
        state = kwargs["request_state"]
        # Use or modify state as needed
        if "counter" not in state:
            state["counter"] = 0
        state["counter"] += 1
        print(f"Callback handler event count: {state['counter']}")

agent = Agent(callback_handler=custom_callback_handler)

result = agent("Hi there!")

print(result.state)

```

The request state:

- Is initialized at the beginning of each agent call
- Persists through recursive event loop cycles
- Can be modified by callback handlers
- Is returned in the AgentResult object

## Session Management

A session represents all of the stateful information that is needed by an agent to function. Strands provides built-in session persistence capabilities that allow agents to maintain state and conversation history across multiple interactions.

```
from strands import Agent
from strands.session.file_session_manager import FileSessionManager

# Create a session manager with a unique session ID
session_manager = FileSessionManager(session_id="test-session")

# Create an agent with the session manager
agent = Agent(session_manager=session_manager)

# Use the agent - all messages and state are automatically persisted
agent("hello")  # This is persisted

```

### Built-in Session Persistence

Strands offers two built-in session managers for persisting agent state and conversation history:

1. **FileSessionManager**: Stores sessions in the local filesystem
1. **S3SessionManager**: Stores sessions in Amazon S3 buckets

#### Using FileSessionManager

The `FileSessionManager` provides a simple way to persist agent sessions to the local filesystem:

```
from strands import Agent
from strands.session.file_session_manager import FileSessionManager

# Create a session manager with a unique session ID
session_manager = FileSessionManager(
    session_id="user-123",
    storage_dir="/path/to/sessions"  # Optional, defaults to a temp directory
)

# Create an agent with the session manager
agent = Agent(session_manager=session_manager)

# Use the agent normally - state and messages will be persisted automatically
agent("Hello, I'm a new user!")

# Later, create a new agent instance with the same session manager
# to continue the conversation where it left off
new_agent = Agent(session_manager=session_manager)
new_agent("Do you remember what I said earlier?")

```

#### Using S3SessionManager

For cloud-based persistence, especially in distributed environments, use the `S3SessionManager`:

```
from strands import Agent
from strands.session.s3_session_manager import S3SessionManager

# Create a session manager that stores data in S3
session_manager = S3SessionManager(
    session_id="user-456",
    bucket="my-agent-sessions",
    prefix="production/",  # Optional key prefix
    region_name="us-west-2"  # Optional AWS region
)

# Create an agent with the session manager
agent = Agent(session_manager=session_manager)

# Use the agent normally - state and messages will be persisted to S3
agent("Tell me about AWS S3")

# Later, even on a different server, create a new agent instance
# with the same session manager to continue the conversation
new_agent = Agent(session_manager=session_manager)
new_agent("Can you elaborate on what you told me about S3?")

```

### How Session Persistence Works

The session persistence system:

1. **Automatically captures**:

   - Agent initialization events
   - Message additions to conversation history
   - Agent state changes

1. **Stores data in a structured format**:

   - `Session`: Top-level container with metadata
   - `SessionAgent`: Agent-specific data including state
   - `SessionMessage`: Individual messages with metadata

1. **Handles serialization**:

   - Properly encodes/decodes complex data types
   - Special handling for binary data using base64 encoding
   - Maintains timestamps for creation and updates

#### File Storage Structure

When using `FileSessionManager`, sessions are stored in the following directory structure:

```
/<sessions_dir>/
└── session_<session_id>/
    ├── session.json                # Session metadata
    └── agents/
        └── agent_<agent_id>/
            ├── agent.json          # Agent state
            └── messages/
                ├── message_<id1>.json
                └── message_<id2>.json

```

When using `S3SessionManager`, a similar structure is maintained using S3 object keys.

### Custom Session Repositories

For advanced use cases, you can implement your own session storage backend by creating a custom session repository:

```
from typing import Optional
from strands import Agent
from strands.session.agent_session_manager import AgentSessionManager
from strands.session.session_repository import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage

class CustomSessionRepository(SessionRepository):
    """Custom session repository implementation."""

    def __init__(self):
        """Initialize with your custom storage backend."""
        # Initialize your storage backend (e.g., database connection)
        self.db = YourDatabaseClient()

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        self.db.sessions.insert(asdict(session))
        return session

    def read_session(self, session_id: str) -> Optional[Session]:
        """Read a session by ID."""
        data = self.db.sessions.find_one({"session_id": session_id})
        if data:
            return Session.from_dict(data)
        return None

    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Create a new Agent in a Session."""
        self.db.agents.insert({
            "session_id": session_id,
            "agent_id": session_agent.agent_id,
            **asdict(session_agent)
        })

    def read_agent(self, session_id: str, agent_id: str) -> Optional[SessionAgent]:
        """Read an Agent."""
        data = self.db.agents.find_one({"session_id": session_id, "agent_id": agent_id})
        if data:
            return SessionAgent.from_dict(data)
        return None

    def update_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Update an Agent."""
        self.db.agents.update(
            {"session_id": session_id, "agent_id": session_agent.agent_id},
            asdict(session_agent)
        )

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Create a new Message for the Agent."""
        self.db.messages.insert({
            "session_id": session_id,
            "agent_id": agent_id,
            "message_id": session_message.message_id,
            **asdict(session_message)
        })

    def read_message(self, session_id: str, agent_id: str, message_id: str) -> Optional[SessionMessage]:
        """Read a Message."""
        data = self.db.messages.find_one({
            "session_id": session_id,
            "agent_id": agent_id,
            "message_id": message_id
        })
        if data:
            return SessionMessage.from_dict(data)
        return None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Update a Message."""
        self.db.messages.update(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "message_id": session_message.message_id
            },
            asdict(session_message)
        )

    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> list[SessionMessage]:
        """List Messages from an Agent with pagination."""
        query = {"session_id": session_id, "agent_id": agent_id}
        cursor = self.db.messages.find(query).sort("created_at").skip(offset)
        if limit:
            cursor = cursor.limit(limit)
        return [SessionMessage.from_dict(msg) for msg in cursor]

# Use your custom repository with AgentSessionManager
custom_repo = CustomSessionRepository()
session_manager = AgentSessionManager(
    session_id="user-789",
    session_repository=custom_repo
)

agent = Agent(session_manager=session_manager)

```

This approach allows you to store session data in any backend system while leveraging the built-in session management logic.

### Session Persistence Best Practices

When implementing session persistence in your applications, consider these best practices:

1. **Use Unique Session IDs**: Generate unique session IDs for each user or conversation context to prevent data overlap.

1. **Consider Storage Requirements**:

   - `FileSessionManager` is ideal for development, testing, or single-server deployments
   - `S3SessionManager` is better for production, distributed systems, or when high availability is required

1. **Security Considerations**:

   - For `FileSessionManager`, ensure the storage directory has appropriate file permissions
   - For `S3SessionManager`, use IAM roles with least privilege and consider server-side encryption
   - Be mindful of storing sensitive user data in sessions

1. **Performance Optimization**:

   - Session data is loaded and saved automatically, so be mindful of large state objects
   - Consider implementing a caching layer for frequently accessed sessions

1. **Error Handling**:

   - Handle `SessionException` errors that might occur during session operations
   - Implement fallback mechanisms for when session storage is unavailable

1. **Session Cleanup**:

   - Implement a strategy for cleaning up old or inactive sessions
   - Consider adding TTL (Time To Live) for sessions in production environments

1. **Testing**:

   - Use `FileSessionManager` with a temporary directory for unit tests
   - Create mock implementations of `SessionRepository` for testing complex scenarios

By following these practices, you can build robust applications that maintain user context effectively across multiple interactions.
