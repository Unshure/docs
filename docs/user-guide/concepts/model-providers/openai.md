# OpenAI

[OpenAI](https://platform.openai.com/docs/overview) is an AI research and deployment company that provides a suite of powerful language models. The Strands Agents SDK implements an OpenAI provider, allowing you to run agents against any OpenAI or OpenAI-compatible model.

## Installation

OpenAI is configured as an optional dependency in Strands Agents. To install, run:

```bash
pip install 'strands-agents[openai]'
```

## Usage

After installing `openai`, you can import and initialize the Strands Agents' OpenAI provider as follows:

```python
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import calculator

model = OpenAIModel(
    client_args={
        "api_key": "<KEY>",
    },
    # **model_config
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

agent = Agent(model=model, tools=[calculator])
response = agent("What is 2+2")
print(response)
```

To connect to a custom OpenAI-compatible server, you will pass in its `base_url` into the `client_args`:

```python
model = OpenAIModel(
    client_args={
      "api_key": "<KEY>",
      "base_url": "<URL>",
    },
    ...
)
```

## Configuration

### Client Configuration

The `client_args` configure the underlying OpenAI client. For a complete list of available arguments, please refer to the OpenAI [source](https://github.com/openai/openai-python).

### Model Configuration

The `model_config` configures the underlying model selected for inference. The supported configurations are:

|  Parameter | Description | Example | Options |
|------------|-------------|---------|---------|
| `model_id` | ID of a model to use | `gpt-4o` | [reference](https://platform.openai.com/docs/models)
| `params` | Model specific parameters | `{"max_tokens": 1000, "temperature": 0.7}` | [reference](https://platform.openai.com/docs/api-reference/chat/create)

## Troubleshooting

### Module Not Found

If you encounter the error `ModuleNotFoundError: No module named 'openai'`, this means you haven't installed the `openai` dependency in your environment. To fix, run `pip install 'strands-agents[openai]'`.

## Structured Output

OpenAI models support structured output through their native tool calling capabilities. When you use [`Agent.structured_output()`](../../../api-reference/agent.md#strands.agent.agent.Agent.structured_output), the Strands SDK automatically converts your Pydantic models to OpenAI's function calling format.

### Usage

```python
from pydantic import BaseModel, Field
from strands import Agent
from strands.models.openai import OpenAIModel

class PersonInfo(BaseModel):
    """Extract person information from text."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")

model = OpenAIModel(
    client_args={"api_key": "<KEY>"},
    model_id="gpt-4o",
    params={"temperature": 0.1}  # Lower temperature for more consistent structured output
)

agent = Agent(model=model)

# Extract structured information
result = agent.structured_output(
    PersonInfo,
    "John Smith is a 30-year-old software engineer working at a tech startup."
)

print(f"Name: {result.name}")      # "John Smith"
print(f"Age: {result.age}")        # 30
print(f"Job: {result.occupation}") # "software engineer"
```

### Complex Structured Output

OpenAI models handle complex nested structures well:

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None

class Contact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class Person(BaseModel):
    """Complete person profile."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    address: Address = Field(description="Home address")
    contacts: List[Contact] = Field(default_factory=list, description="Contact information")
    skills: List[str] = Field(default_factory=list, description="Professional skills")

# Use with complex data
result = agent.structured_output(
    Person,
    """
    Extract information about Sarah Johnson, a 28-year-old data scientist.
    She lives at 456 Oak Avenue, San Francisco, CA 94102.
    Her email is sarah.j@email.com and phone is (555) 123-4567.
    She specializes in machine learning and Python programming.
    """
)

print(f"Name: {result.name}")
print(f"City: {result.address.city}")
print(f"Email: {result.contacts[0].email}")
print(f"Skills: {result.skills}")
```

### Best Practices for OpenAI Structured Output

1. **Use descriptive field descriptions**: OpenAI models benefit from clear field descriptions
2. **Set appropriate temperature**: Lower temperatures (0.0-0.3) work better for structured output
3. **Handle validation errors**: Implement proper error handling for malformed responses
4. **Model selection**: GPT-4 models generally perform better than GPT-3.5 for complex structured output

### Supported Models

All OpenAI models that support function calling work with structured output:

- GPT-4 and GPT-4 Turbo models
- GPT-4o and GPT-4o-mini
- GPT-3.5-turbo (with some limitations on complex schemas)

For the most up-to-date list of supported models, see the [OpenAI documentation](https://platform.openai.com/docs/guides/function-calling).

### Error Handling

```python
from pydantic import ValidationError

try:
    result = agent.structured_output(PersonInfo, "Extract info from this text...")
except ValidationError as e:
    print(f"Structured output validation failed: {e}")
    # Handle the error appropriately
except Exception as e:
    print(f"OpenAI API error: {e}")
    # Handle API errors
```

## References

- [API](../../../api-reference/models.md)
- [OpenAI](https://platform.openai.com/docs/overview)
