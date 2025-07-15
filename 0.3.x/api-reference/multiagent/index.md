# `strands.multiagent`

Multiagent capabilities for Strands Agents.

This module provides support for multiagent systems, including agent-to-agent (A2A) communication protocols and coordination mechanisms.

Submodules

a2a: Implementation of the Agent-to-Agent (A2A) protocol, which enables standardized communication between agents.

## `strands.multiagent.base`

Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).

### `MultiAgentBase`

Bases: `ABC`

Base class for multi-agent helpers.

This class integrates with existing Strands Agent instances and provides multi-agent orchestration capabilities.

Source code in `strands/multiagent/base.py`

```
class MultiAgentBase(ABC):
    """Base class for multi-agent helpers.

    This class integrates with existing Strands Agent instances and provides
    multi-agent orchestration capabilities.
    """

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    async def execute_async(self, task: str) -> MultiAgentResult:
        """Execute task asynchronously."""
        raise NotImplementedError("execute_async not implemented")

    @abstractmethod
    # TODO: for task - multi-modal input (Message), list of messages
    def execute(self, task: str) -> MultiAgentResult:
        """Execute task synchronously."""
        raise NotImplementedError("execute not implemented")

```

#### `execute(task)`

Execute task synchronously.

Source code in `strands/multiagent/base.py`

```
@abstractmethod
# TODO: for task - multi-modal input (Message), list of messages
def execute(self, task: str) -> MultiAgentResult:
    """Execute task synchronously."""
    raise NotImplementedError("execute not implemented")

```

#### `execute_async(task)`

Execute task asynchronously.

Source code in `strands/multiagent/base.py`

```
@abstractmethod
# TODO: for task - multi-modal input (Message), list of messages
async def execute_async(self, task: str) -> MultiAgentResult:
    """Execute task asynchronously."""
    raise NotImplementedError("execute_async not implemented")

```

### `MultiAgentResult`

Result from multi-agent execution with accumulated metrics.

Source code in `strands/multiagent/base.py`

```
@dataclass
class MultiAgentResult:
    """Result from multi-agent execution with accumulated metrics."""

    results: dict[str, NodeResult]
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0

```

### `NodeResult`

Unified result from node execution - handles both Agent and nested MultiAgentBase results.

The status field represents the semantic outcome of the node's work:

- COMPLETED: The node's task was successfully accomplished
- FAILED: The node's task failed or produced an error

Source code in `strands/multiagent/base.py`

```
@dataclass
class NodeResult:
    """Unified result from node execution - handles both Agent and nested MultiAgentBase results.

    The status field represents the semantic outcome of the node's work:
    - COMPLETED: The node's task was successfully accomplished
    - FAILED: The node's task failed or produced an error
    """

    # Core result data - single AgentResult, nested MultiAgentResult, or Exception
    result: Union[AgentResult, "MultiAgentResult", Exception]

    # Execution metadata
    execution_time: int = 0
    status: Status = Status.PENDING

    # Accumulated metrics from this node and all children
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0

    def get_agent_results(self) -> list[AgentResult]:
        """Get all AgentResult objects from this node, flattened if nested."""
        if isinstance(self.result, Exception):
            return []  # No agent results for exceptions
        elif isinstance(self.result, AgentResult):
            return [self.result]
        else:
            # Flatten nested results from MultiAgentResult
            flattened = []
            for nested_node_result in self.result.results.values():
                flattened.extend(nested_node_result.get_agent_results())
            return flattened

```

#### `get_agent_results()`

Get all AgentResult objects from this node, flattened if nested.

Source code in `strands/multiagent/base.py`

```
def get_agent_results(self) -> list[AgentResult]:
    """Get all AgentResult objects from this node, flattened if nested."""
    if isinstance(self.result, Exception):
        return []  # No agent results for exceptions
    elif isinstance(self.result, AgentResult):
        return [self.result]
    else:
        # Flatten nested results from MultiAgentResult
        flattened = []
        for nested_node_result in self.result.results.values():
            flattened.extend(nested_node_result.get_agent_results())
        return flattened

```

### `Status`

Bases: `Enum`

Execution status for both graphs and nodes.

Source code in `strands/multiagent/base.py`

```
class Status(Enum):
    """Execution status for both graphs and nodes."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

```

## `strands.multiagent.graph`

Directed Acyclic Graph (DAG) Multi-Agent Pattern Implementation.

This module provides a deterministic DAG-based agent orchestration system where agents or MultiAgentBase instances (like Swarm or Graph) are nodes in a graph, executed according to edge dependencies, with output from one node passed as input to connected nodes.

Key Features:

- Agents and MultiAgentBase instances (Swarm, Graph, etc.) as graph nodes
- Deterministic execution order based on DAG structure
- Output propagation along edges
- Topological sort for execution ordering
- Clear dependency management
- Supports nested graphs (Graph as a node in another Graph)

### `Graph`

Bases: `MultiAgentBase`

Directed Acyclic Graph multi-agent orchestration.

Source code in `strands/multiagent/graph.py`

```
class Graph(MultiAgentBase):
    """Directed Acyclic Graph multi-agent orchestration."""

    def __init__(self, nodes: dict[str, GraphNode], edges: set[GraphEdge], entry_points: set[GraphNode]) -> None:
        """Initialize Graph."""
        super().__init__()

        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.state = GraphState()

    def execute(self, task: str) -> GraphResult:
        """Execute task synchronously."""

        def execute() -> GraphResult:
            return asyncio.run(self.execute_async(task))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def execute_async(self, task: str) -> GraphResult:
        """Execute the graph asynchronously."""
        logger.debug("task=<%s> | starting graph execution", task)

        # Initialize state
        self.state = GraphState(
            status=Status.EXECUTING,
            task=task,
            total_nodes=len(self.nodes),
            edges=[(edge.from_node, edge.to_node) for edge in self.edges],
            entry_points=list(self.entry_points),
        )

        start_time = time.time()
        try:
            await self._execute_graph()
            self.state.status = Status.COMPLETED
            logger.debug("status=<%s> | graph execution completed", self.state.status)

        except Exception:
            logger.exception("graph execution failed")
            self.state.status = Status.FAILED
            raise
        finally:
            self.state.execution_time = round((time.time() - start_time) * 1000)

        return self._build_result()

    async def _execute_graph(self) -> None:
        """Unified execution flow with conditional routing."""
        ready_nodes = list(self.entry_points)

        while ready_nodes:
            current_batch = ready_nodes.copy()
            ready_nodes.clear()

            # Execute current batch of ready nodes
            for node in current_batch:
                if node not in self.state.completed_nodes:
                    await self._execute_node(node)

                    # Find newly ready nodes after this execution
                    ready_nodes.extend(self._find_newly_ready_nodes())

    def _find_newly_ready_nodes(self) -> list["GraphNode"]:
        """Find nodes that became ready after the last execution."""
        newly_ready = []
        for _node_id, node in self.nodes.items():
            if (
                node not in self.state.completed_nodes
                and node not in self.state.failed_nodes
                and self._is_node_ready_with_conditions(node)
            ):
                newly_ready.append(node)
        return newly_ready

    def _is_node_ready_with_conditions(self, node: GraphNode) -> bool:
        """Check if a node is ready considering conditional edges."""
        # Get incoming edges to this node
        incoming_edges = [edge for edge in self.edges if edge.to_node == node]

        if not incoming_edges:
            return node in self.entry_points

        # Check if at least one incoming edge condition is satisfied
        for edge in incoming_edges:
            if edge.from_node in self.state.completed_nodes:
                if edge.should_traverse(self.state):
                    logger.debug(
                        "from=<%s>, to=<%s> | edge ready via satisfied condition", edge.from_node.node_id, node.node_id
                    )
                    return True
                else:
                    logger.debug(
                        "from=<%s>, to=<%s> | edge condition not satisfied", edge.from_node.node_id, node.node_id
                    )
        return False

    async def _execute_node(self, node: GraphNode) -> None:
        """Execute a single node with error handling."""
        node.execution_status = Status.EXECUTING
        logger.debug("node_id=<%s> | executing node", node.node_id)

        start_time = time.time()
        try:
            # Build node input from satisfied dependencies
            node_input = self._build_node_input(node)

            # Execute based on node type and create unified NodeResult
            if isinstance(node.executor, MultiAgentBase):
                multi_agent_result = await node.executor.execute_async(node_input)

                # Create NodeResult with MultiAgentResult directly
                node_result = NodeResult(
                    result=multi_agent_result,  # type is MultiAgentResult
                    execution_time=multi_agent_result.execution_time,
                    status=Status.COMPLETED,
                    accumulated_usage=multi_agent_result.accumulated_usage,
                    accumulated_metrics=multi_agent_result.accumulated_metrics,
                    execution_count=multi_agent_result.execution_count,
                )

            elif isinstance(node.executor, Agent):
                agent_response: AgentResult | None = (
                    None  # Initialize with None to handle case where no result is yielded
                )
                async for event in node.executor.stream_async(node_input):
                    if "result" in event:
                        agent_response = cast(AgentResult, event["result"])

                if not agent_response:
                    raise ValueError(f"Node '{node.node_id}' did not return a result")

                # Extract metrics from agent response
                usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
                metrics = Metrics(latencyMs=0)
                if hasattr(agent_response, "metrics") and agent_response.metrics:
                    if hasattr(agent_response.metrics, "accumulated_usage"):
                        usage = agent_response.metrics.accumulated_usage
                    if hasattr(agent_response.metrics, "accumulated_metrics"):
                        metrics = agent_response.metrics.accumulated_metrics

                node_result = NodeResult(
                    result=agent_response,  # type is AgentResult
                    execution_time=round((time.time() - start_time) * 1000),
                    status=Status.COMPLETED,
                    accumulated_usage=usage,
                    accumulated_metrics=metrics,
                    execution_count=1,
                )
            else:
                raise ValueError(f"Node '{node.node_id}' of type '{type(node.executor)}' is not supported")

            # Mark as completed
            node.execution_status = Status.COMPLETED
            node.result = node_result
            node.execution_time = node_result.execution_time
            self.state.completed_nodes.add(node)
            self.state.results[node.node_id] = node_result
            self.state.execution_order.append(node)

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            logger.debug(
                "node_id=<%s>, execution_time=<%dms> | node completed successfully", node.node_id, node.execution_time
            )

        except Exception as e:
            logger.error("node_id=<%s>, error=<%s> | node failed", node.node_id, e)
            execution_time = round((time.time() - start_time) * 1000)

            # Create a NodeResult for the failed node
            node_result = NodeResult(
                result=e,  # Store exception as result
                execution_time=execution_time,
                status=Status.FAILED,
                accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                accumulated_metrics=Metrics(latencyMs=execution_time),
                execution_count=1,
            )

            node.execution_status = Status.FAILED
            node.result = node_result
            node.execution_time = execution_time
            self.state.failed_nodes.add(node)
            self.state.results[node.node_id] = node_result  # Store in results for consistency

            raise

    def _accumulate_metrics(self, node_result: NodeResult) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)
        self.state.execution_count += node_result.execution_count

    def _build_node_input(self, node: GraphNode) -> str:
        """Build input text for a node based on dependency outputs."""
        # Get satisfied dependencies
        dependency_results = {}
        for edge in self.edges:
            if (
                edge.to_node == node
                and edge.from_node in self.state.completed_nodes
                and edge.from_node.node_id in self.state.results
            ):
                if edge.should_traverse(self.state):
                    dependency_results[edge.from_node.node_id] = self.state.results[edge.from_node.node_id]

        if not dependency_results:
            return self.state.task

        # Combine task with dependency outputs
        input_parts = [f"Original Task: {self.state.task}", "\nInputs from previous nodes:"]

        for dep_id, node_result in dependency_results.items():
            input_parts.append(f"\nFrom {dep_id}:")
            # Get all agent results from this node (flattened if nested)
            agent_results = node_result.get_agent_results()
            for result in agent_results:
                agent_name = getattr(result, "agent_name", "Agent")
                result_text = str(result)
                input_parts.append(f"  - {agent_name}: {result_text}")

        return "\n".join(input_parts)

    def _build_result(self) -> GraphResult:
        """Build graph result from current state."""
        return GraphResult(
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=self.state.execution_count,
            execution_time=self.state.execution_time,
            status=self.state.status,
            total_nodes=self.state.total_nodes,
            completed_nodes=len(self.state.completed_nodes),
            failed_nodes=len(self.state.failed_nodes),
            execution_order=self.state.execution_order,
            edges=self.state.edges,
            entry_points=self.state.entry_points,
        )

```

#### `__init__(nodes, edges, entry_points)`

Initialize Graph.

Source code in `strands/multiagent/graph.py`

```
def __init__(self, nodes: dict[str, GraphNode], edges: set[GraphEdge], entry_points: set[GraphNode]) -> None:
    """Initialize Graph."""
    super().__init__()

    self.nodes = nodes
    self.edges = edges
    self.entry_points = entry_points
    self.state = GraphState()

```

#### `execute(task)`

Execute task synchronously.

Source code in `strands/multiagent/graph.py`

```
def execute(self, task: str) -> GraphResult:
    """Execute task synchronously."""

    def execute() -> GraphResult:
        return asyncio.run(self.execute_async(task))

    with ThreadPoolExecutor() as executor:
        future = executor.submit(execute)
        return future.result()

```

#### `execute_async(task)`

Execute the graph asynchronously.

Source code in `strands/multiagent/graph.py`

```
async def execute_async(self, task: str) -> GraphResult:
    """Execute the graph asynchronously."""
    logger.debug("task=<%s> | starting graph execution", task)

    # Initialize state
    self.state = GraphState(
        status=Status.EXECUTING,
        task=task,
        total_nodes=len(self.nodes),
        edges=[(edge.from_node, edge.to_node) for edge in self.edges],
        entry_points=list(self.entry_points),
    )

    start_time = time.time()
    try:
        await self._execute_graph()
        self.state.status = Status.COMPLETED
        logger.debug("status=<%s> | graph execution completed", self.state.status)

    except Exception:
        logger.exception("graph execution failed")
        self.state.status = Status.FAILED
        raise
    finally:
        self.state.execution_time = round((time.time() - start_time) * 1000)

    return self._build_result()

```

### `GraphBuilder`

Builder pattern for constructing graphs.

Source code in `strands/multiagent/graph.py`

```
class GraphBuilder:
    """Builder pattern for constructing graphs."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with empty collections."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: set[GraphEdge] = set()
        self.entry_points: set[GraphNode] = set()

    def add_node(self, executor: Agent | MultiAgentBase, node_id: str | None = None) -> GraphNode:
        """Add an Agent or MultiAgentBase instance as a node to the graph."""
        # Auto-generate node_id if not provided
        if node_id is None:
            node_id = getattr(executor, "id", None) or getattr(executor, "name", None) or f"node_{len(self.nodes)}"

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        node = GraphNode(node_id=node_id, executor=executor)
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        from_node: str | GraphNode,
        to_node: str | GraphNode,
        condition: Callable[[GraphState], bool] | None = None,
    ) -> GraphEdge:
        """Add an edge between two nodes with optional condition function that receives full GraphState."""

        def resolve_node(node: str | GraphNode, node_type: str) -> GraphNode:
            if isinstance(node, str):
                if node not in self.nodes:
                    raise ValueError(f"{node_type} node '{node}' not found")
                return self.nodes[node]
            else:
                if node not in self.nodes.values():
                    raise ValueError(f"{node_type} node object has not been added to the graph, use graph.add_node")
                return node

        from_node_obj = resolve_node(from_node, "Source")
        to_node_obj = resolve_node(to_node, "Target")

        # Add edge and update dependencies
        edge = GraphEdge(from_node=from_node_obj, to_node=to_node_obj, condition=condition)
        self.edges.add(edge)
        to_node_obj.dependencies.add(from_node_obj)
        return edge

    def set_entry_point(self, node_id: str) -> "GraphBuilder":
        """Set a node as an entry point for graph execution."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self.entry_points.add(self.nodes[node_id])
        return self

    def build(self) -> "Graph":
        """Build and validate the graph."""
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")

        # Auto-detect entry points if none specified
        if not self.entry_points:
            self.entry_points = {node for node_id, node in self.nodes.items() if not node.dependencies}
            logger.debug(
                "entry_points=<%s> | auto-detected entrypoints", ", ".join(node.node_id for node in self.entry_points)
            )
            if not self.entry_points:
                raise ValueError("No entry points found - all nodes have dependencies")

        # Validate entry points and check for cycles
        self._validate_graph()

        return Graph(nodes=self.nodes.copy(), edges=self.edges.copy(), entry_points=self.entry_points.copy())

    def _validate_graph(self) -> None:
        """Validate graph structure and detect cycles."""
        # Validate entry points exist
        entry_point_ids = {node.node_id for node in self.entry_points}
        invalid_entries = entry_point_ids - set(self.nodes.keys())
        if invalid_entries:
            raise ValueError(f"Entry points not found in nodes: {invalid_entries}")

        # Check for cycles using DFS with color coding
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}

        def has_cycle_from(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if colors[node_id] == BLACK:
                return False

            colors[node_id] = GRAY
            # Check all outgoing edges for cycles
            for edge in self.edges:
                if edge.from_node.node_id == node_id and has_cycle_from(edge.to_node.node_id):
                    return True
            colors[node_id] = BLACK
            return False

        # Check for cycles from each unvisited node
        if any(colors[node_id] == WHITE and has_cycle_from(node_id) for node_id in self.nodes):
            raise ValueError("Graph contains cycles - must be a directed acyclic graph")

```

#### `__init__()`

Initialize GraphBuilder with empty collections.

Source code in `strands/multiagent/graph.py`

```
def __init__(self) -> None:
    """Initialize GraphBuilder with empty collections."""
    self.nodes: dict[str, GraphNode] = {}
    self.edges: set[GraphEdge] = set()
    self.entry_points: set[GraphNode] = set()

```

#### `add_edge(from_node, to_node, condition=None)`

Add an edge between two nodes with optional condition function that receives full GraphState.

Source code in `strands/multiagent/graph.py`

```
def add_edge(
    self,
    from_node: str | GraphNode,
    to_node: str | GraphNode,
    condition: Callable[[GraphState], bool] | None = None,
) -> GraphEdge:
    """Add an edge between two nodes with optional condition function that receives full GraphState."""

    def resolve_node(node: str | GraphNode, node_type: str) -> GraphNode:
        if isinstance(node, str):
            if node not in self.nodes:
                raise ValueError(f"{node_type} node '{node}' not found")
            return self.nodes[node]
        else:
            if node not in self.nodes.values():
                raise ValueError(f"{node_type} node object has not been added to the graph, use graph.add_node")
            return node

    from_node_obj = resolve_node(from_node, "Source")
    to_node_obj = resolve_node(to_node, "Target")

    # Add edge and update dependencies
    edge = GraphEdge(from_node=from_node_obj, to_node=to_node_obj, condition=condition)
    self.edges.add(edge)
    to_node_obj.dependencies.add(from_node_obj)
    return edge

```

#### `add_node(executor, node_id=None)`

Add an Agent or MultiAgentBase instance as a node to the graph.

Source code in `strands/multiagent/graph.py`

```
def add_node(self, executor: Agent | MultiAgentBase, node_id: str | None = None) -> GraphNode:
    """Add an Agent or MultiAgentBase instance as a node to the graph."""
    # Auto-generate node_id if not provided
    if node_id is None:
        node_id = getattr(executor, "id", None) or getattr(executor, "name", None) or f"node_{len(self.nodes)}"

    if node_id in self.nodes:
        raise ValueError(f"Node '{node_id}' already exists")

    node = GraphNode(node_id=node_id, executor=executor)
    self.nodes[node_id] = node
    return node

```

#### `build()`

Build and validate the graph.

Source code in `strands/multiagent/graph.py`

```
def build(self) -> "Graph":
    """Build and validate the graph."""
    if not self.nodes:
        raise ValueError("Graph must contain at least one node")

    # Auto-detect entry points if none specified
    if not self.entry_points:
        self.entry_points = {node for node_id, node in self.nodes.items() if not node.dependencies}
        logger.debug(
            "entry_points=<%s> | auto-detected entrypoints", ", ".join(node.node_id for node in self.entry_points)
        )
        if not self.entry_points:
            raise ValueError("No entry points found - all nodes have dependencies")

    # Validate entry points and check for cycles
    self._validate_graph()

    return Graph(nodes=self.nodes.copy(), edges=self.edges.copy(), entry_points=self.entry_points.copy())

```

#### `set_entry_point(node_id)`

Set a node as an entry point for graph execution.

Source code in `strands/multiagent/graph.py`

```
def set_entry_point(self, node_id: str) -> "GraphBuilder":
    """Set a node as an entry point for graph execution."""
    if node_id not in self.nodes:
        raise ValueError(f"Node '{node_id}' not found")
    self.entry_points.add(self.nodes[node_id])
    return self

```

### `GraphEdge`

Represents an edge in the graph with an optional condition.

Source code in `strands/multiagent/graph.py`

```
@dataclass
class GraphEdge:
    """Represents an edge in the graph with an optional condition."""

    from_node: "GraphNode"
    to_node: "GraphNode"
    condition: Callable[[GraphState], bool] | None = None

    def __hash__(self) -> int:
        """Return hash for GraphEdge based on from_node and to_node."""
        return hash((self.from_node.node_id, self.to_node.node_id))

    def should_traverse(self, state: GraphState) -> bool:
        """Check if this edge should be traversed based on condition."""
        if self.condition is None:
            return True
        return self.condition(state)

```

#### `__hash__()`

Return hash for GraphEdge based on from_node and to_node.

Source code in `strands/multiagent/graph.py`

```
def __hash__(self) -> int:
    """Return hash for GraphEdge based on from_node and to_node."""
    return hash((self.from_node.node_id, self.to_node.node_id))

```

#### `should_traverse(state)`

Check if this edge should be traversed based on condition.

Source code in `strands/multiagent/graph.py`

```
def should_traverse(self, state: GraphState) -> bool:
    """Check if this edge should be traversed based on condition."""
    if self.condition is None:
        return True
    return self.condition(state)

```

### `GraphNode`

Represents a node in the graph.

The execution_status tracks the node's lifecycle within graph orchestration:

- PENDING: Node hasn't started executing yet
- EXECUTING: Node is currently running
- COMPLETED/FAILED: Node finished executing (regardless of result quality)

Source code in `strands/multiagent/graph.py`

```
@dataclass
class GraphNode:
    """Represents a node in the graph.

    The execution_status tracks the node's lifecycle within graph orchestration:
    - PENDING: Node hasn't started executing yet
    - EXECUTING: Node is currently running
    - COMPLETED/FAILED: Node finished executing (regardless of result quality)
    """

    node_id: str
    executor: Agent | MultiAgentBase
    dependencies: set["GraphNode"] = field(default_factory=set)
    execution_status: Status = Status.PENDING
    result: NodeResult | None = None
    execution_time: int = 0

    def __hash__(self) -> int:
        """Return hash for GraphNode based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        """Return equality for GraphNode based on node_id."""
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id

```

#### `__eq__(other)`

Return equality for GraphNode based on node_id.

Source code in `strands/multiagent/graph.py`

```
def __eq__(self, other: Any) -> bool:
    """Return equality for GraphNode based on node_id."""
    if not isinstance(other, GraphNode):
        return False
    return self.node_id == other.node_id

```

#### `__hash__()`

Return hash for GraphNode based on node_id.

Source code in `strands/multiagent/graph.py`

```
def __hash__(self) -> int:
    """Return hash for GraphNode based on node_id."""
    return hash(self.node_id)

```

### `GraphResult`

Bases: `MultiAgentResult`

Result from graph execution - extends MultiAgentResult with graph-specific details.

The status field represents the outcome of the graph execution:

- COMPLETED: The graph execution was successfully accomplished
- FAILED: The graph execution failed or produced an error

Source code in `strands/multiagent/graph.py`

```
@dataclass
class GraphResult(MultiAgentResult):
    """Result from graph execution - extends MultiAgentResult with graph-specific details.

    The status field represents the outcome of the graph execution:
    - COMPLETED: The graph execution was successfully accomplished
    - FAILED: The graph execution failed or produced an error
    """

    status: Status = Status.PENDING
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    execution_order: list["GraphNode"] = field(default_factory=list)
    edges: list[Tuple["GraphNode", "GraphNode"]] = field(default_factory=list)
    entry_points: list["GraphNode"] = field(default_factory=list)

```

### `GraphState`

Graph execution state.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `status` | `Status` | Current execution status of the graph. | | `completed_nodes` | `set[GraphNode]` | Set of nodes that have completed execution. | | `failed_nodes` | `set[GraphNode]` | Set of nodes that failed during execution. | | `execution_order` | `list[GraphNode]` | List of nodes in the order they were executed. | | `task` | `str` | The original input prompt/query provided to the graph execution. This represents the actual work to be performed by the graph as a whole. Entry point nodes receive this task as their input if they have no dependencies. |

Source code in `strands/multiagent/graph.py`

```
@dataclass
class GraphState:
    """Graph execution state.

    Attributes:
        status: Current execution status of the graph.
        completed_nodes: Set of nodes that have completed execution.
        failed_nodes: Set of nodes that failed during execution.
        execution_order: List of nodes in the order they were executed.
        task: The original input prompt/query provided to the graph execution.
              This represents the actual work to be performed by the graph as a whole.
              Entry point nodes receive this task as their input if they have no dependencies.
    """

    # Execution state
    status: Status = Status.PENDING
    completed_nodes: set["GraphNode"] = field(default_factory=set)
    failed_nodes: set["GraphNode"] = field(default_factory=set)
    execution_order: list["GraphNode"] = field(default_factory=list)
    task: str = ""

    # Results
    results: dict[str, NodeResult] = field(default_factory=dict)

    # Accumulated metrics
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0

    # Graph structure info
    total_nodes: int = 0
    edges: list[Tuple["GraphNode", "GraphNode"]] = field(default_factory=list)
    entry_points: list["GraphNode"] = field(default_factory=list)

```

## `strands.multiagent.a2a`

Agent-to-Agent (A2A) communication protocol implementation for Strands Agents.

This module provides classes and utilities for enabling Strands Agents to communicate with other agents using the Agent-to-Agent (A2A) protocol.

Docs: https://google-a2a.github.io/A2A/latest/

Classes:

| Name | Description | | --- | --- | | `A2AAgent` | A wrapper that adapts a Strands Agent to be A2A-compatible. |

### `strands.multiagent.a2a.executor`

Strands Agent executor for the A2A protocol.

This module provides the StrandsA2AExecutor class, which adapts a Strands Agent to be used as an executor in the A2A protocol. It handles the execution of agent requests and the conversion of Strands Agent streamed responses to A2A events.

The A2A AgentExecutor ensures clients recieve responses for synchronous and streamed requests to the A2AServer.

#### `StrandsA2AExecutor`

Bases: `AgentExecutor`

Executor that adapts a Strands Agent to the A2A protocol.

This executor uses streaming mode to handle the execution of agent requests and converts Strands Agent responses to A2A protocol events.

Source code in `strands/multiagent/a2a/executor.py`

```
class StrandsA2AExecutor(AgentExecutor):
    """Executor that adapts a Strands Agent to the A2A protocol.

    This executor uses streaming mode to handle the execution of agent requests
    and converts Strands Agent responses to A2A protocol events.
    """

    def __init__(self, agent: SAAgent):
        """Initialize a StrandsA2AExecutor.

        Args:
            agent: The Strands Agent instance to adapt to the A2A protocol.
        """
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a request using the Strands Agent and send the response as A2A events.

        This method executes the user's input using the Strands Agent in streaming mode
        and converts the agent's response to A2A events.

        Args:
            context: The A2A request context, containing the user's input and task metadata.
            event_queue: The A2A event queue used to send response events back to the client.

        Raises:
            ServerError: If an error occurs during agent execution
        """
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            await self._execute_streaming(context, updater)
        except Exception as e:
            raise ServerError(error=InternalError()) from e

    async def _execute_streaming(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Execute request in streaming mode.

        Streams the agent's response in real-time, sending incremental updates
        as they become available from the agent.

        Args:
            context: The A2A request context, containing the user's input and other metadata.
            updater: The task updater for managing task state and sending updates.
        """
        logger.info("Executing request in streaming mode")
        user_input = context.get_user_input()
        try:
            async for event in self.agent.stream_async(user_input):
                await self._handle_streaming_event(event, updater)
        except Exception:
            logger.exception("Error in streaming execution")
            raise

    async def _handle_streaming_event(self, event: dict[str, Any], updater: TaskUpdater) -> None:
        """Handle a single streaming event from the Strands Agent.

        Processes streaming events from the agent, converting data chunks to A2A
        task updates and handling the final result when streaming is complete.

        Args:
            event: The streaming event from the agent, containing either 'data' for
                incremental content or 'result' for the final response.
            updater: The task updater for managing task state and sending updates.
        """
        logger.debug("Streaming event: %s", event)
        if "data" in event:
            if text_content := event["data"]:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        text_content,
                        updater.context_id,
                        updater.task_id,
                    ),
                )
        elif "result" in event:
            await self._handle_agent_result(event["result"], updater)
        else:
            logger.warning("Unexpected streaming event: %s", event)

    async def _handle_agent_result(self, result: SAAgentResult | None, updater: TaskUpdater) -> None:
        """Handle the final result from the Strands Agent.

        Processes the agent's final result, extracts text content from the response,
        and adds it as an artifact to the task before marking the task as complete.

        Args:
            result: The agent result object containing the final response, or None if no result.
            updater: The task updater for managing task state and adding the final artifact.
        """
        if final_content := str(result):
            await updater.add_artifact(
                [Part(root=TextPart(text=final_content))],
                name="agent_response",
            )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        This method is called when a request cancellation is requested. Currently,
        cancellation is not supported by the Strands Agent executor, so this method
        always raises an UnsupportedOperationError.

        Args:
            context: The A2A request context.
            event_queue: The A2A event queue.

        Raises:
            ServerError: Always raised with an UnsupportedOperationError, as cancellation
                is not currently supported.
        """
        logger.warning("Cancellation requested but not supported")
        raise ServerError(error=UnsupportedOperationError())

```

##### `__init__(agent)`

Initialize a StrandsA2AExecutor.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `Agent` | The Strands Agent instance to adapt to the A2A protocol. | *required* |

Source code in `strands/multiagent/a2a/executor.py`

```
def __init__(self, agent: SAAgent):
    """Initialize a StrandsA2AExecutor.

    Args:
        agent: The Strands Agent instance to adapt to the A2A protocol.
    """
    self.agent = agent

```

##### `cancel(context, event_queue)`

Cancel an ongoing execution.

This method is called when a request cancellation is requested. Currently, cancellation is not supported by the Strands Agent executor, so this method always raises an UnsupportedOperationError.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `context` | `RequestContext` | The A2A request context. | *required* | | `event_queue` | `EventQueue` | The A2A event queue. | *required* |

Raises:

| Type | Description | | --- | --- | | `ServerError` | Always raised with an UnsupportedOperationError, as cancellation is not currently supported. |

Source code in `strands/multiagent/a2a/executor.py`

```
async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
    """Cancel an ongoing execution.

    This method is called when a request cancellation is requested. Currently,
    cancellation is not supported by the Strands Agent executor, so this method
    always raises an UnsupportedOperationError.

    Args:
        context: The A2A request context.
        event_queue: The A2A event queue.

    Raises:
        ServerError: Always raised with an UnsupportedOperationError, as cancellation
            is not currently supported.
    """
    logger.warning("Cancellation requested but not supported")
    raise ServerError(error=UnsupportedOperationError())

```

##### `execute(context, event_queue)`

Execute a request using the Strands Agent and send the response as A2A events.

This method executes the user's input using the Strands Agent in streaming mode and converts the agent's response to A2A events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `context` | `RequestContext` | The A2A request context, containing the user's input and task metadata. | *required* | | `event_queue` | `EventQueue` | The A2A event queue used to send response events back to the client. | *required* |

Raises:

| Type | Description | | --- | --- | | `ServerError` | If an error occurs during agent execution |

Source code in `strands/multiagent/a2a/executor.py`

```
async def execute(
    self,
    context: RequestContext,
    event_queue: EventQueue,
) -> None:
    """Execute a request using the Strands Agent and send the response as A2A events.

    This method executes the user's input using the Strands Agent in streaming mode
    and converts the agent's response to A2A events.

    Args:
        context: The A2A request context, containing the user's input and task metadata.
        event_queue: The A2A event queue used to send response events back to the client.

    Raises:
        ServerError: If an error occurs during agent execution
    """
    task = context.current_task
    if not task:
        task = new_task(context.message)  # type: ignore
        await event_queue.enqueue_event(task)

    updater = TaskUpdater(event_queue, task.id, task.contextId)

    try:
        await self._execute_streaming(context, updater)
    except Exception as e:
        raise ServerError(error=InternalError()) from e

```

### `strands.multiagent.a2a.server`

A2A-compatible wrapper for Strands Agent.

This module provides the A2AAgent class, which adapts a Strands Agent to the A2A protocol, allowing it to be used in A2A-compatible systems.

#### `A2AServer`

A2A-compatible wrapper for Strands Agent.

Source code in `strands/multiagent/a2a/server.py`

```
class A2AServer:
    """A2A-compatible wrapper for Strands Agent."""

    def __init__(
        self,
        agent: SAAgent,
        *,
        # AgentCard
        host: str = "0.0.0.0",
        port: int = 9000,
        version: str = "0.0.1",
        skills: list[AgentSkill] | None = None,
    ):
        """Initialize an A2A-compatible server from a Strands agent.

        Args:
            agent: The Strands Agent to wrap with A2A compatibility.
            host: The hostname or IP address to bind the A2A server to. Defaults to "0.0.0.0".
            port: The port to bind the A2A server to. Defaults to 9000.
            version: The version of the agent. Defaults to "0.0.1".
            skills: The list of capabilities or functions the agent can perform.
        """
        self.host = host
        self.port = port
        self.http_url = f"http://{self.host}:{self.port}/"
        self.version = version
        self.strands_agent = agent
        self.name = self.strands_agent.name
        self.description = self.strands_agent.description
        self.capabilities = AgentCapabilities(streaming=True)
        self.request_handler = DefaultRequestHandler(
            agent_executor=StrandsA2AExecutor(self.strands_agent),
            task_store=InMemoryTaskStore(),
        )
        self._agent_skills = skills
        logger.info("Strands' integration with A2A is experimental. Be aware of frequent breaking changes.")

    @property
    def public_agent_card(self) -> AgentCard:
        """Get the public AgentCard for this agent.

        The AgentCard contains metadata about the agent, including its name,
        description, URL, version, skills, and capabilities. This information
        is used by other agents and systems to discover and interact with this agent.

        Returns:
            AgentCard: The public agent card containing metadata about this agent.

        Raises:
            ValueError: If name or description is None or empty.
        """
        if not self.name:
            raise ValueError("A2A agent name cannot be None or empty")
        if not self.description:
            raise ValueError("A2A agent description cannot be None or empty")

        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.http_url,
            version=self.version,
            skills=self.agent_skills,
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=self.capabilities,
        )

    def _get_skills_from_tools(self) -> list[AgentSkill]:
        """Get the list of skills from Strands agent tools.

        Skills represent specific capabilities that the agent can perform.
        Strands agent tools are adapted to A2A skills.

        Returns:
            list[AgentSkill]: A list of skills this agent provides.
        """
        return [
            AgentSkill(name=config["name"], id=config["name"], description=config["description"], tags=[])
            for config in self.strands_agent.tool_registry.get_all_tools_config().values()
        ]

    @property
    def agent_skills(self) -> list[AgentSkill]:
        """Get the list of skills this agent provides."""
        return self._agent_skills if self._agent_skills is not None else self._get_skills_from_tools()

    @agent_skills.setter
    def agent_skills(self, skills: list[AgentSkill]) -> None:
        """Set the list of skills this agent provides.

        Args:
            skills: A list of AgentSkill objects to set for this agent.
        """
        self._agent_skills = skills

    def to_starlette_app(self) -> Starlette:
        """Create a Starlette application for serving this agent via HTTP.

        This method creates a Starlette application that can be used to serve
        the agent via HTTP using the A2A protocol.

        Returns:
            Starlette: A Starlette application configured to serve this agent.
        """
        return A2AStarletteApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

    def to_fastapi_app(self) -> FastAPI:
        """Create a FastAPI application for serving this agent via HTTP.

        This method creates a FastAPI application that can be used to serve
        the agent via HTTP using the A2A protocol.

        Returns:
            FastAPI: A FastAPI application configured to serve this agent.
        """
        return A2AFastAPIApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

    def serve(self, app_type: Literal["fastapi", "starlette"] = "starlette", **kwargs: Any) -> None:
        """Start the A2A server with the specified application type.

        This method starts an HTTP server that exposes the agent via the A2A protocol.
        The server can be implemented using either FastAPI or Starlette, depending on
        the specified app_type.

        Args:
            app_type: The type of application to serve, either "fastapi" or "starlette".
                Defaults to "starlette".
            **kwargs: Additional keyword arguments to pass to uvicorn.run.
        """
        try:
            logger.info("Starting Strands A2A server...")
            if app_type == "fastapi":
                uvicorn.run(self.to_fastapi_app(), host=self.host, port=self.port, **kwargs)
            else:
                uvicorn.run(self.to_starlette_app(), host=self.host, port=self.port, **kwargs)
        except KeyboardInterrupt:
            logger.warning("Strands A2A server shutdown requested (KeyboardInterrupt).")
        except Exception:
            logger.exception("Strands A2A server encountered exception.")
        finally:
            logger.info("Strands A2A server has shutdown.")

```

##### `agent_skills`

Get the list of skills this agent provides.

##### `public_agent_card`

Get the public AgentCard for this agent.

The AgentCard contains metadata about the agent, including its name, description, URL, version, skills, and capabilities. This information is used by other agents and systems to discover and interact with this agent.

Returns:

| Name | Type | Description | | --- | --- | --- | | `AgentCard` | `AgentCard` | The public agent card containing metadata about this agent. |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If name or description is None or empty. |

##### `__init__(agent, *, host='0.0.0.0', port=9000, version='0.0.1', skills=None)`

Initialize an A2A-compatible server from a Strands agent.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `Agent` | The Strands Agent to wrap with A2A compatibility. | *required* | | `host` | `str` | The hostname or IP address to bind the A2A server to. Defaults to "0.0.0.0". | `'0.0.0.0'` | | `port` | `int` | The port to bind the A2A server to. Defaults to 9000. | `9000` | | `version` | `str` | The version of the agent. Defaults to "0.0.1". | `'0.0.1'` | | `skills` | `list[AgentSkill] | None` | The list of capabilities or functions the agent can perform. | `None` |

Source code in `strands/multiagent/a2a/server.py`

```
def __init__(
    self,
    agent: SAAgent,
    *,
    # AgentCard
    host: str = "0.0.0.0",
    port: int = 9000,
    version: str = "0.0.1",
    skills: list[AgentSkill] | None = None,
):
    """Initialize an A2A-compatible server from a Strands agent.

    Args:
        agent: The Strands Agent to wrap with A2A compatibility.
        host: The hostname or IP address to bind the A2A server to. Defaults to "0.0.0.0".
        port: The port to bind the A2A server to. Defaults to 9000.
        version: The version of the agent. Defaults to "0.0.1".
        skills: The list of capabilities or functions the agent can perform.
    """
    self.host = host
    self.port = port
    self.http_url = f"http://{self.host}:{self.port}/"
    self.version = version
    self.strands_agent = agent
    self.name = self.strands_agent.name
    self.description = self.strands_agent.description
    self.capabilities = AgentCapabilities(streaming=True)
    self.request_handler = DefaultRequestHandler(
        agent_executor=StrandsA2AExecutor(self.strands_agent),
        task_store=InMemoryTaskStore(),
    )
    self._agent_skills = skills
    logger.info("Strands' integration with A2A is experimental. Be aware of frequent breaking changes.")

```

##### `serve(app_type='starlette', **kwargs)`

Start the A2A server with the specified application type.

This method starts an HTTP server that exposes the agent via the A2A protocol. The server can be implemented using either FastAPI or Starlette, depending on the specified app_type.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `app_type` | `Literal['fastapi', 'starlette']` | The type of application to serve, either "fastapi" or "starlette". Defaults to "starlette". | `'starlette'` | | `**kwargs` | `Any` | Additional keyword arguments to pass to uvicorn.run. | `{}` |

Source code in `strands/multiagent/a2a/server.py`

```
def serve(self, app_type: Literal["fastapi", "starlette"] = "starlette", **kwargs: Any) -> None:
    """Start the A2A server with the specified application type.

    This method starts an HTTP server that exposes the agent via the A2A protocol.
    The server can be implemented using either FastAPI or Starlette, depending on
    the specified app_type.

    Args:
        app_type: The type of application to serve, either "fastapi" or "starlette".
            Defaults to "starlette".
        **kwargs: Additional keyword arguments to pass to uvicorn.run.
    """
    try:
        logger.info("Starting Strands A2A server...")
        if app_type == "fastapi":
            uvicorn.run(self.to_fastapi_app(), host=self.host, port=self.port, **kwargs)
        else:
            uvicorn.run(self.to_starlette_app(), host=self.host, port=self.port, **kwargs)
    except KeyboardInterrupt:
        logger.warning("Strands A2A server shutdown requested (KeyboardInterrupt).")
    except Exception:
        logger.exception("Strands A2A server encountered exception.")
    finally:
        logger.info("Strands A2A server has shutdown.")

```

##### `to_fastapi_app()`

Create a FastAPI application for serving this agent via HTTP.

This method creates a FastAPI application that can be used to serve the agent via HTTP using the A2A protocol.

Returns:

| Name | Type | Description | | --- | --- | --- | | `FastAPI` | `FastAPI` | A FastAPI application configured to serve this agent. |

Source code in `strands/multiagent/a2a/server.py`

```
def to_fastapi_app(self) -> FastAPI:
    """Create a FastAPI application for serving this agent via HTTP.

    This method creates a FastAPI application that can be used to serve
    the agent via HTTP using the A2A protocol.

    Returns:
        FastAPI: A FastAPI application configured to serve this agent.
    """
    return A2AFastAPIApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

```

##### `to_starlette_app()`

Create a Starlette application for serving this agent via HTTP.

This method creates a Starlette application that can be used to serve the agent via HTTP using the A2A protocol.

Returns:

| Name | Type | Description | | --- | --- | --- | | `Starlette` | `Starlette` | A Starlette application configured to serve this agent. |

Source code in `strands/multiagent/a2a/server.py`

```
def to_starlette_app(self) -> Starlette:
    """Create a Starlette application for serving this agent via HTTP.

    This method creates a Starlette application that can be used to serve
    the agent via HTTP using the A2A protocol.

    Returns:
        Starlette: A Starlette application configured to serve this agent.
    """
    return A2AStarletteApplication(agent_card=self.public_agent_card, http_handler=self.request_handler).build()

```
