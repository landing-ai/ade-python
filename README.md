<!-- @format -->

<div align="center">

<img src="PATH_TO_LOGO" alt="LandingAI" width="420" />

**Agentic Document Extraction Python Library**

[![PyPI version](https://img.shields.io/pypi/v/landingai-ade.svg?label=pypi%20\(stable\))](https://pypi.org/project/landingai-ade/)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License](https://img.shields.io/pypi/l/landingai-ade)](https://pypi.org/project/landingai-ade/)


**[Playground](https://va.landing.ai) · [Discord](https://discord.com/invite/RVcW3j9RgR) · [Blog](https://landing.ai/blog) · [Docs](https://docs.landing.ai/ade/ade-overview)**

</div>


A Python library for interacting with the **LandingAI Agentic Document Extraction REST API**, designed for flexibility, reliability, clarity, and performance. Built for Python 3.9+ and generated with [Stainless](https://www.stainless.com/).


## ✨ Features

* ✅ Fully-typed SDK with Pydantic response models
* ⚡️ Sync & Async clients
* 📄 Large document processing via async jobs
* 🔁 Built-in retries with exponential backoff
* 🔐 Secure API key handling
* 📦 Seamless file uploads
* 🧩 Schema-based data extraction
* 🔌 Pluggable HTTP backends (`httpx` or `aiohttp`)


## 📚 Documentation

* REST API Docs: [https://docs.landing.ai/](https://docs.landing.ai/)
* Full SDK Reference: [api.md](api.md)


## 🚀 Installation

```bash
pip install landingai-ade
```

Optional (for enhanced async performance):

```bash
pip install landingai-ade[aiohttp]
```


## 🔧 Quick Start

### Basic Parse

```python
import os
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE(
    apikey=os.environ.get("VISION_AGENT_API_KEY"),
    environment="eu",  # defaults to "production"
)

response = client.parse(
    document=Path("path/to/file"),
    model="dpt-2-latest",
)

print(response.chunks)
```

> 💡 Tip: Use `python-dotenv` and a `.env` file to avoid committing API keys.


## 🧵 Asynchronous Processing

### Async Parse Jobs

```python
from pathlib import Path
from landingai_ade import LandingAIADE

client = LandingAIADE(apikey=os.environ.get("VISION_AGENT_API_KEY"))

job = client.parse_jobs.create(
    document=Path("large_file.pdf"),
    model="dpt-2-latest",
)

print(f"Job ID: {job.job_id}")

status = client.parse_jobs.get(job.job_id)
print(f"Status: {status.status}")
```


## 🧠 Structured Extraction

```python
from pathlib import Path
from pydantic import BaseModel, Field
from landingai_ade import LandingAIADE
from landingai_ade.lib import pydantic_to_json_schema

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

schema = pydantic_to_json_schema(Person)

client = LandingAIADE()
response = client.extract(
    schema=schema,
    markdown=Path("file.md"),
)
```


## ⚙️ Async Client Usage

```python
import asyncio
from pathlib import Path
from landingai_ade import AsyncLandingAIADE

async def main():
    client = AsyncLandingAIADE()
    response = await client.parse(
        document=Path("file.pdf"),
        model="dpt-2-latest",
    )
    print(response.chunks)

asyncio.run(main())
```

### With aiohttp Backend

```python
from landingai_ade import AsyncLandingAIADE, DefaultAioHttpClient

async with AsyncLandingAIADE(http_client=DefaultAioHttpClient()) as client:
    response = await client.parse(...)
```


## 🧾 File Uploads

Supported types:

* `bytes`
* `PathLike`
* Tuple: `(filename, content, media-type)`

```python
client.parse(document=Path("/path/to/file"))
```


## ❗ Error Handling

```python
import landingai_ade

try:
    client.parse()
except landingai_ade.APIConnectionError:
    print("Connection error")
except landingai_ade.RateLimitError:
    print("Too many requests")
except landingai_ade.APIStatusError as e:
    print(e.status_code)
```

### Error Codes

| HTTP Code | Error Type               |
| --------- | ------------------------ |
| 400       | BadRequestError          |
| 401       | AuthenticationError      |
| 403       | PermissionDeniedError    |
| 404       | NotFoundError            |
| 422       | UnprocessableEntityError |
| 429       | RateLimitError           |
| >=500     | InternalServerError      |
| N/A       | APIConnectionError       |


## 🔄 Retry Logic

Retries apply to:

* 408, 409, 429, and >=500 responses
* Network failures

```python
client = LandingAIADE(max_retries=0)
client.with_options(max_retries=5).parse()
```


## ⏱ Timeouts

```python
import httpx

client = LandingAIADE(timeout=httpx.Timeout(60.0, read=5.0, write=10.0))
client.with_options(timeout=5.0).parse()
```


## 📈 Logging

Enable via environment variable:

```bash
export LANDINGAI_ADE_LOG=info
```

Options: `info`, `debug`


## 🔍 Raw & Streaming Responses

```python
with client.with_streaming_response.parse() as response:
    for line in response.iter_lines():
        print(line)
```


## 🧩 Advanced Usage

### Custom Requests

```python
response = client.post(
    "/foo",
    cast_to=httpx.Response,
    body={"my_param": True},
)
```

### Undocumented Fields

```python
print(response.unknown_field)
print(response.model_extra)
```


## 🔧 HTTP Client Configuration

```python
from landingai_ade import DefaultHttpxClient

client = LandingAIADE(
    http_client=DefaultHttpxClient(proxy="http://proxy"),
)
```


## 📦 Version Info

```python
import landingai_ade
print(landingai_ade.__version__)
```

Follows SemVer with careful backward compatibility.


## 🛠 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

Feel free to open issues for bugs, suggestions, or feature requests:
[https://github.com/landing-ai/ade-python/issues](https://github.com/landing-ai/ade-python/issues)


## ✅ Requirements

* Python 3.9 or higher

---

Built with ❤️ by LandingAI
