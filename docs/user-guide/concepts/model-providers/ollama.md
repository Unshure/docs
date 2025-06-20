# Ollama

Ollama is a framework for running open-source large language models locally. Strands provides native support for Ollama, allowing you to use locally-hosted models in your agents.

The [`OllamaModel`](../../../api-reference/models.md#strands.models.ollama) class in Strands enables seamless integration with Ollama's API, supporting:

- Text generation
- Image understanding
- Tool/function calling
- Streaming responses
- Configuration management

## Getting Started

### Prerequisites

First install the python client into your python environment:
```bash
pip install 'strands-agents[ollama]'
```

Next, you'll need to install and setup ollama itself.

#### Option 1: Native Installation

1. Install Ollama by following the instructions at [ollama.ai](https://ollama.ai)
2. Pull your desired model:
   ```bash
   ollama pull llama3
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```

#### Option 2: Docker Installation

1. Pull the Ollama Docker image:
   ```bash
   docker pull ollama/ollama
   ```

2. Run the Ollama container:
   ```bash
   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```

   > Note: Add `--gpus=all` if you have a GPU and if Docker GPU support is configured.

3. Pull a model using the Docker container:
   ```bash
   docker exec -it ollama ollama pull llama3
   ```

4. Verify the Ollama server is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Basic Usage

Here's how to create an agent using an Ollama model:

```python
from strands import Agent
from strands.models.ollama import OllamaModel

# Create an Ollama model instance
ollama_model = OllamaModel(
    host="http://localhost:11434",  # Ollama server address
    model_id="llama3"               # Specify which model to use
)

# Create an agent using the Ollama model
agent = Agent(model=ollama_model)

# Use the agent
agent("Tell me about Strands agents.") # Prints model output to stdout by default
```

## Configuration Options

The [`OllamaModel`](../../../api-reference/models.md#strands.models.ollama) supports various [configuration parameters](../../../api-reference/models.md#strands.models.ollama.OllamaModel.OllamaConfig):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `host` | The address of the Ollama server | Required |
| `model_id` | The Ollama model identifier | Required |
| `keep_alive` | How long the model stays loaded in memory | "5m" |
| `max_tokens` | Maximum number of tokens to generate | None |
| `temperature` | Controls randomness (higher = more random) | None |
| `top_p` | Controls diversity via nucleus sampling | None |
| `stop_sequences` | List of sequences that stop generation | None |
| `options` | Additional model parameters (e.g., top_k) | None |
| `additional_args` | Any additional arguments for the request | None |

### Example with Configuration

```python
from strands import Agent
from strands.models.ollama import OllamaModel

# Create a configured Ollama model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3",
    temperature=0.7,
    keep_alive="10m",
    stop_sequences=["###", "END"],
    options={"top_k": 40}
)

# Create an agent with the configured model
agent = Agent(model=ollama_model)

# Use the agent
response = agent("Write a short story about an AI assistant.")
```

## Advanced Features

### Updating Configuration at Runtime

You can update the model configuration during runtime:

```python
# Create the model with initial configuration
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3",
    temperature=0.7
)

# Update configuration later
ollama_model.update_config(
    temperature=0.9,
    top_p=0.8
)
```

This is especially useful if you want a tool to update the model's config for you:

```python
@tool
def update_model_id(model_id: str, agent: Agent) -> str:
    """
    Update the model id of the agent

    Args:
      model_id: Ollama model id to use.
    """
    print(f"Updating model_id to {model_id}")
    agent.model.update_config(model_id=model_id)
    return f"Model updated to {model_id}"


@tool
def update_temperature(temperature: float, agent: Agent) -> str:
    """
    Update the temperature of the agent

    Args:
      temperature: Temperature value for the model to use.
    """
    print(f"Updating Temperature to {temperature}")
    agent.model.update_config(temperature=temperature)
    return f"Temperature updated to {temperature}"
```

### Using Different Models

Ollama supports many different models. You can switch between them (make sure they are pulled first). See the list of
available models here: https://ollama.com/search

```python
# Create models for different use cases
creative_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3",
    temperature=0.8
)

factual_model = OllamaModel(
    host="http://localhost:11434",
    model_id="mistral",
    temperature=0.2
)

# Create agents with different models
creative_agent = Agent(model=creative_model)
factual_agent = Agent(model=factual_model)
```

## Tool Support

[Ollama models that support tool use](https://ollama.com/search?c=tools) can use tools through Strands's tool system:

```python
from strands import Agent
from strands.models.ollama import OllamaModel
from strands_tools import calculator, current_time

# Create an Ollama model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3"
)

# Create an agent with tools
agent = Agent(
    model=ollama_model,
    tools=[calculator, current_time]
)

# Use the agent with tools
response = agent("What's the square root of 144 plus the current time?")
```

## Troubleshooting

### Common Issues

1. **Connection Refused**:
    - Ensure the Ollama server is running (`ollama serve` or check Docker container status)
    - Verify the host URL is correct
    - For Docker: Check if port 11434 is properly exposed

2. **Model Not Found**:
    - Pull the model first: `ollama pull model_name` or `docker exec -it ollama ollama pull model_name`
    - Check for typos in the model_id

3. **Module Not Found**:
    - If you encounter the error `ModuleNotFoundError: No module named 'ollama'`, this means you haven't installed the `ollama` dependency in your python environment
    - To fix, run `pip install 'strands-agents[ollama]'`

## Structured Output

Ollama supports structured output for models that have tool calling capabilities. When you use [`Agent.structured_output()`](../../../api-reference/agent.md#strands.agent.agent.Agent.structured_output), the Strands SDK converts your Pydantic models to tool specifications that compatible Ollama models can understand.

### Supported Models

Not all Ollama models support structured output. Models that support tool calling include:

- **Llama 3.1 and 3.2 models**: `llama3.1:8b`, `llama3.1:70b`, `llama3.2:3b`
- **Mistral models**: `mistral:7b`, `mistral-nemo:12b`
- **Qwen models**: `qwen2.5:7b`, `qwen2.5:14b`
- **Phi models**: `phi3:14b`

Check the [Ollama models page](https://ollama.com/search?c=tools) for the latest list of tool-capable models.

### Usage

```python
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.ollama import OllamaModel

class CodeAnalysis(BaseModel):
    """Analyze code structure and quality."""
    language: str = Field(description="Programming language")
    complexity: str = Field(description="Code complexity: low, medium, high")
    issues: List[str] = Field(description="Potential issues or improvements")
    score: int = Field(description="Code quality score 1-10", ge=1, le=10)

# Use a model that supports tool calling
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.1:8b",  # Tool-capable model
    temperature=0.1  # Low temperature for consistent structured output
)

agent = Agent(model=ollama_model)

# Extract structured code analysis
result = agent.structured_output(
    CodeAnalysis,
    """
    Analyze this Python function:
    
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    
    This is a recursive implementation of the Fibonacci sequence.
    """
)

print(f"Language: {result.language}")
print(f"Complexity: {result.complexity}")
print(f"Issues: {result.issues}")
print(f"Score: {result.score}")
```

### Local Model Advantages

Using Ollama for structured output offers unique benefits:

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class LocalDataAnalysis(BaseModel):
    """Analyze sensitive data locally without external API calls."""
    data_type: str = Field(description="Type of data being analyzed")
    patterns: List[str] = Field(description="Identified patterns")
    anomalies: List[str] = Field(description="Unusual or anomalous entries")
    privacy_score: int = Field(description="Privacy compliance score 1-10", ge=1, le=10)
    recommendations: List[str] = Field(description="Recommendations for improvement")

# Process sensitive data locally
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.1:8b",
    temperature=0.0
)

agent = Agent(model=ollama_model)

# Analyze sensitive data without sending to external APIs
result = agent.structured_output(
    LocalDataAnalysis,
    "Analyze this customer database for patterns and privacy compliance..."
)
```

### Performance Considerations

Local models have different performance characteristics:

```python
# Optimize for speed vs quality
fast_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",  # Smaller, faster model
    temperature=0.0,
    num_ctx=2048  # Smaller context for faster processing
)

# Optimize for quality
quality_model = OllamaModel(
    host="http://localhost:11434", 
    model_id="llama3.1:70b",  # Larger, more capable model
    temperature=0.1,
    num_ctx=4096  # Larger context for complex tasks
)
```

### Best Practices for Ollama Structured Output

1. **Choose appropriate models**: Use tool-capable models like Llama 3.1/3.2 or Mistral
2. **Optimize for hardware**: Smaller models (3B-8B) for limited resources, larger (70B+) for better quality
3. **Use low temperature**: Set temperature to 0.0-0.2 for consistent structured output
4. **Handle timeouts**: Local models may be slower than cloud APIs
5. **Monitor resources**: Large models require significant RAM and compute
6. **Test model capabilities**: Not all models handle complex schemas equally well

### Error Handling

```python
from pydantic import ValidationError
import requests

try:
    result = agent.structured_output(CodeAnalysis, "Analyze this code...")
except ValidationError as e:
    print(f"Structured output validation failed: {e}")
    # The model's response didn't match the expected schema
except requests.exceptions.ConnectionError:
    print("Cannot connect to Ollama server. Is it running?")
except requests.exceptions.Timeout:
    print("Request timed out. Consider using a smaller model or increasing timeout.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Model Recommendations

**For Simple Structured Output:**
- `llama3.2:3b` - Fast, good for basic extraction
- `qwen2.5:7b` - Balanced speed and quality

**For Complex Structured Output:**
- `llama3.1:8b` - Good balance of capability and speed
- `llama3.1:70b` - Best quality, requires significant resources

**For Code Analysis:**
- `mistral:7b` - Good at code understanding
- `qwen2.5:14b` - Strong technical analysis capabilities

### Limitations

- **Model dependency**: Only models with tool calling support work with structured output
- **Performance**: Local models are typically slower than cloud APIs
- **Resource requirements**: Larger models need significant RAM and compute
- **Quality variance**: Smaller models may produce less consistent structured output
- **Schema complexity**: Very complex schemas may challenge smaller local models

## Related Resources

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama)
- [Available Ollama Models](https://ollama.ai/library)
