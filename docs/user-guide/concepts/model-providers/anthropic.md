# Anthropic

[Anthropic](https://docs.anthropic.com/en/home) is an AI safety and research company focused on building reliable, interpretable, and steerable AI systems. Included in their offerings is the Claude AI family of models, which are known for their conversational abilities, careful reasoning, and capacity to follow complex instructions. The Strands Agents SDK implements an Anthropic provider, allowing users to run agents against Claude models directly.

## Installation

Anthropic is configured as an optional dependency in Strands. To install, run:

```bash
pip install 'strands-agents[anthropic]'
```

## Usage

After installing `anthropic`, you can import and initialize Strands' Anthropic provider as follows:

```python
from strands import Agent
from strands.models.anthropic import AnthropicModel
from strands_tools import calculator

model = AnthropicModel(
    client_args={
        "api_key": "<KEY>",
    },
    # **model_config
    max_tokens=1028,
    model_id="claude-3-7-sonnet-20250219",
    params={
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
```

## Configuration

### Client Configuration

The `client_args` configure the underlying Anthropic client. For a complete list of available arguments, please refer to the Anthropic [docs](https://docs.anthropic.com/en/api/client-sdks).

### Model Configuration

The `model_config` configures the underlying model selected for inference. The supported configurations are:

|  Parameter | Description | Example | Options |
|------------|-------------|---------|---------|
| `max_tokens` | Maximum number of tokens to generate before stopping | `1028` | [reference](https://docs.anthropic.com/en/api/messages#body-max-tokens)
| `model_id` | ID of a model to use | `claude-3-7-sonnet-20250219` | [reference](https://docs.anthropic.com/en/api/messages#body-model)
| `params` | Model specific parameters | `{"max_tokens": 1000, "temperature": 0.7}` | [reference](https://docs.anthropic.com/en/api/messages)

## Troubleshooting

### Module Not Found

If you encounter the error `ModuleNotFoundError: No module named 'anthropic'`, this means you haven't installed the `anthropic` dependency in your environment. To fix, run `pip install 'strands-agents[anthropic]'`.

## Structured Output

Anthropic's Claude models support structured output through their tool calling capabilities. When you use [`Agent.structured_output()`](../../../api-reference/agent.md#strands.agent.agent.Agent.structured_output), the Strands SDK converts your Pydantic models to Anthropic's tool specification format.

### Usage

```python
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.anthropic import AnthropicModel

class BookAnalysis(BaseModel):
    """Analyze a book's key information."""
    title: str = Field(description="The book's title")
    author: str = Field(description="The book's author")
    genre: str = Field(description="Primary genre or category")
    summary: str = Field(description="Brief summary of the book")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)

model = AnthropicModel(
    client_args={"api_key": "<KEY>"},
    max_tokens=1024,
    model_id="claude-3-7-sonnet-20250219",
    params={"temperature": 0.2}  # Lower temperature for consistent structured output
)

agent = Agent(model=model)

# Extract structured book information
result = agent.structured_output(
    BookAnalysis,
    """
    Analyze this book: "The Hitchhiker's Guide to the Galaxy" by Douglas Adams.
    It's a science fiction comedy about Arthur Dent's adventures through space
    after Earth is destroyed. It's widely considered a classic of humorous sci-fi.
    """
)

print(f"Title: {result.title}")
print(f"Author: {result.author}")
print(f"Genre: {result.genre}")
print(f"Rating: {result.rating}")
```

### Advanced Structured Output

Claude models excel at complex reasoning and structured analysis:

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Argument(BaseModel):
    point: str = Field(description="The main argument point")
    evidence: List[str] = Field(description="Supporting evidence")
    strength: int = Field(description="Argument strength 1-5", ge=1, le=5)

class DebateAnalysis(BaseModel):
    """Comprehensive analysis of a debate or discussion."""
    topic: str = Field(description="Main topic being debated")
    arguments_for: List[Argument] = Field(description="Arguments supporting the position")
    arguments_against: List[Argument] = Field(description="Arguments opposing the position")
    conclusion: str = Field(description="Balanced conclusion")
    confidence: float = Field(description="Confidence in analysis 0-1", ge=0, le=1)

# Analyze complex debates
result = agent.structured_output(
    DebateAnalysis,
    """
    Analyze the debate about remote work vs office work.
    Consider productivity, collaboration, work-life balance, company culture,
    and employee satisfaction. Provide a balanced analysis.
    """
)

print(f"Topic: {result.topic}")
print(f"Arguments for: {len(result.arguments_for)}")
print(f"Arguments against: {len(result.arguments_against)}")
print(f"Conclusion: {result.conclusion}")
```

### Best Practices for Anthropic Structured Output

1. **Leverage Claude's reasoning**: Claude excels at complex analysis and reasoning tasks
2. **Use detailed descriptions**: Provide clear field descriptions for better results
3. **Optimize temperature**: Use 0.0-0.3 for consistent structured output
4. **Handle long content**: Claude can process longer texts effectively for structured extraction
5. **Utilize validation**: Take advantage of Pydantic's validation features

### Supported Models

All recent Claude models support structured output through tool calling:

- Claude 3.5 Sonnet
- Claude 3.5 Haiku  
- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku

For the latest model availability, see the [Anthropic documentation](https://docs.anthropic.com/en/docs/about-claude/models).

### Model-Specific Considerations

**Claude 3.5 Sonnet**: Best overall performance for structured output with complex reasoning
**Claude 3.5 Haiku**: Fastest response times, good for simple structured extraction
**Claude 3 Opus**: Highest capability for very complex structured analysis

### Error Handling

```python
from pydantic import ValidationError
from anthropic import APIError

try:
    result = agent.structured_output(BookAnalysis, "Analyze this book...")
except ValidationError as e:
    print(f"Structured output validation failed: {e}")
    # The model's response didn't match the expected schema
except APIError as e:
    print(f"Anthropic API error: {e}")
    # Handle API-specific errors
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Tips

- **Batch processing**: For multiple structured extractions, consider processing them together
- **Context utilization**: Claude can maintain context across multiple structured output calls
- **Schema complexity**: Claude handles complex nested schemas well, but simpler schemas are more reliable

## References

- [API](../../../api-reference/models.md)
- [Anthropic](https://docs.anthropic.com/en/home)

