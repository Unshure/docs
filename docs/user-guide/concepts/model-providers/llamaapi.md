# Llama API

[Llama API](https://llama.developer.meta.com?utm_source=partner-strandsagent&utm_medium=website) is a Meta-hosted API service that helps you integrate Llama models into your applications quickly and efficiently.

Llama API provides access to Llama models through a simple API interface, with inference provided by Meta, so you can focus on building AI-powered solutions without managing your own inference infrastructure.

With Llama API, you get access to state-of-the-art AI capabilities through a developer-friendly interface designed for simplicity and performance.

## Installation

Llama API is configured as an optional dependency in Strands Agents. To install, run:

```bash
pip install 'strands-agents[llamaapi]'
```

## Usage

After installing `llamaapi`, you can import and initialize Strands Agents' Llama API provider as follows:

```python
from strands import Agent
from strands.models.llamaapi import LlamaAPIModel
from strands_tools import calculator

model = LlamaAPIModel(
    client_args={
        "api_key": "<KEY>",
    },
    # **model_config
    model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
```

## Configuration

### Client Configuration

The `client_args` configure the underlying LlamaAPI client. For a complete list of available arguments, please refer to the LlamaAPI [docs](https://llama.developer.meta.com/docs/).


### Model Configuration

The `model_config` configures the underlying model selected for inference. The supported configurations are:

|  Parameter | Description                                                                                         | Example | Options |
|------------|-----------------------------------------------------------------------------------------------------|---------|---------|
| `model_id` | ID of a model to use                                                                                | `Llama-4-Maverick-17B-128E-Instruct-FP8` | [reference](https://llama.developer.meta.com/docs/)
| `repetition_penalty` | Controls the likelihood and generating repetitive responses. (minimum: 1, maximum: 2, default: 1)   |  `1`  | [reference](https://llama.developer.meta.com/docs/api/chat)
| `temperature` | Controls randomness of the response by setting a temperature.                                       | `0.7` | [reference](https://llama.developer.meta.com/docs/api/chat)
| `top_p` | Controls diversity of the response by setting a probability threshold when choosing the next token. | `0.9` | [reference](https://llama.developer.meta.com/docs/api/chat)
| `max_completion_tokens` | The maximum number of tokens to generate.                                                           | `4096` | [reference](https://llama.developer.meta.com/docs/api/chat)
| `top_k` | Only sample from the top K options for each subsequent token.                                       | `10` | [reference](https://llama.developer.meta.com/docs/api/chat)


## Troubleshooting

### Module Not Found

If you encounter the error `ModuleNotFoundError: No module named 'llamaapi'`, this means you haven't installed the `llamaapi` dependency in your environment. To fix, run `pip install 'strands-agents[llamaapi]'`.

## Structured Output

Llama API models support structured output through their tool calling capabilities. When you use [`Agent.structured_output()`](../../../api-reference/agent.md#strands.agent.agent.Agent.structured_output), the Strands SDK converts your Pydantic models to tool specifications that Llama models can understand.

### Usage

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from strands import Agent
from strands.models.llamaapi import LlamaAPIModel

class ResearchSummary(BaseModel):
    """Summarize research findings."""
    topic: str = Field(description="Main research topic")
    key_findings: List[str] = Field(description="Primary research findings")
    methodology: str = Field(description="Research methodology used")
    confidence_level: float = Field(description="Confidence in findings 0-1", ge=0, le=1)
    recommendations: List[str] = Field(description="Actionable recommendations")

model = LlamaAPIModel(
    client_args={"api_key": "<KEY>"},
    model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
    temperature=0.1,  # Low temperature for consistent structured output
    max_completion_tokens=2000
)

agent = Agent(model=model)

# Extract structured research summary
result = agent.structured_output(
    ResearchSummary,
    """
    Analyze this research: A study of 500 remote workers found that 
    productivity increased by 23% when using structured daily schedules.
    The study used time-tracking software and productivity metrics over 6 months.
    Researchers recommend implementing structured work blocks and regular breaks.
    """
)

print(f"Topic: {result.topic}")
print(f"Key Findings: {result.key_findings}")
print(f"Methodology: {result.methodology}")
print(f"Confidence: {result.confidence_level}")
print(f"Recommendations: {result.recommendations}")
```

### Advanced Structured Output

Llama models excel at complex reasoning and analysis tasks:

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class CompetitiveAnalysis(BaseModel):
    """Comprehensive competitive analysis."""
    company_name: str = Field(description="Company being analyzed")
    market_position: str = Field(description="Market position: leader, challenger, follower, niche")
    strengths: List[str] = Field(description="Key competitive strengths")
    weaknesses: List[str] = Field(description="Areas of weakness")
    opportunities: List[str] = Field(description="Market opportunities")
    threats: List[str] = Field(description="Competitive threats")
    overall_score: int = Field(description="Overall competitive score 1-10", ge=1, le=10)

class MarketAnalysis(BaseModel):
    """Complete market analysis with multiple companies."""
    market_name: str = Field(description="Market or industry name")
    market_size: Optional[str] = Field(description="Market size estimate")
    growth_rate: Optional[float] = Field(description="Annual growth rate percentage")
    competitors: List[CompetitiveAnalysis] = Field(description="Competitor analyses")
    market_trends: List[str] = Field(description="Key market trends")

# Analyze complex market data
result = agent.structured_output(
    MarketAnalysis,
    """
    Analyze the cloud computing market. Key players include AWS (market leader with 
    strong infrastructure but high costs), Microsoft Azure (growing fast with 
    enterprise integration), and Google Cloud (innovative but smaller market share).
    The market is worth $500B and growing at 15% annually. Key trends include 
    multi-cloud adoption, edge computing, and AI integration.
    """
)

print(f"Market: {result.market_name}")
print(f"Size: {result.market_size}")
print(f"Growth: {result.growth_rate}%")
print(f"Competitors: {len(result.competitors)}")
for competitor in result.competitors:
    print(f"  - {competitor.company_name}: {competitor.market_position}")
```

### Best Practices for Llama API Structured Output

1. **Use appropriate models**: Larger Llama models generally perform better for complex structured output
2. **Optimize temperature**: Use 0.0-0.3 for consistent structured responses
3. **Provide clear descriptions**: Llama models benefit from detailed field descriptions
4. **Handle token limits**: Monitor `max_completion_tokens` for complex schemas
5. **Leverage reasoning**: Llama models excel at analytical and reasoning tasks
6. **Test thoroughly**: Validate structured output quality with your specific use cases

### Supported Models

Llama API provides access to various Llama model variants. Models that support structured output include:

- **Llama 4 Maverick models**: Latest generation with enhanced tool calling
- **Llama 3.1 models**: Strong structured output capabilities
- **Llama 3.2 models**: Optimized for various tasks including structured generation

Check the [Llama API documentation](https://llama.developer.meta.com/docs/) for the latest available models and their capabilities.

### Configuration for Structured Output

Optimize your Llama API model for structured output:

```python
model = LlamaAPIModel(
    client_args={"api_key": "<KEY>"},
    model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
    temperature=0.1,           # Low temperature for consistency
    max_completion_tokens=3000, # Sufficient tokens for complex structures
    top_p=0.9,                 # Focused sampling
    repetition_penalty=1.1,    # Reduce repetitive responses
    top_k=40                   # Limit token choices for consistency
)
```

### Error Handling

```python
from pydantic import ValidationError
import requests

try:
    result = agent.structured_output(ResearchSummary, "Analyze this research...")
except ValidationError as e:
    print(f"Structured output validation failed: {e}")
    # The model's response didn't match the expected schema
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Tips

- **Batch processing**: For multiple structured extractions, consider processing them in a single request
- **Schema optimization**: Simpler schemas generally produce more reliable results
- **Context management**: Llama models can maintain context across multiple structured output calls
- **Model selection**: Choose the right model size for your complexity needs

### Limitations

- **Model availability**: Structured output support depends on the specific Llama model version
- **Schema complexity**: Very complex nested schemas may challenge smaller models
- **Token limits**: Large structured outputs may approach token limits
- **API rate limits**: Consider rate limiting when making multiple structured output requests

## References

- [API](../../../api-reference/models.md)
- [LlamaAPI](https://llama.developer.meta.com/docs/)
