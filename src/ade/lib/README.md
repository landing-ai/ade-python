# ADE SDK Custom Utilities

This directory contains custom utility functions that extend the base Stainless-generated SDK.

## Available Utilities

### `pydantic_to_json_schema(model)`

Converts a Pydantic BaseModel to a JSON schema string suitable for use with the ADE API.

**Usage:**
```python
from pydantic import BaseModel, Field
from ade.lib import pydantic_to_json_schema
from LandingAIAde import Landingai

# Define your schema
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

# Convert to JSON schema
schema = pydantic_to_json_schema(Person)

# Use with the SDK
client = Landingai(apikey="your-key")
response = client.extract(schema=schema, markdown=content)
```

## Philosophy

We provide minimal utilities that work with the SDK, not wrappers around it. This approach:

- **Transparency**: Users see exactly what API calls are made
- **Control**: Users decide when and how to call the API
- **Simplicity**: Just utility functions, no hidden complexity
- **Flexibility**: Use the utilities with the SDK however you want

## Examples

See `/examples/simple_schema_example.py` for complete examples of using these utilities with the SDK.

## Environment Variables

- `LANDINGAI_API_KEY`: Your Landing AI API key
- `LANDINGAI_ENVIRONMENT`: API environment (default: "production", can be "eu")