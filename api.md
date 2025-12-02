# Shared Types

```python
from landingai_ade.types import ParseGroundingBox, ParseMetadata
```

# LandingAIADE

Types:

```python
from landingai_ade.types import ExtractResponse, ParseResponse, SplitResponse
```

Methods:

- <code title="post /v1/ade/extract">client.<a href="./src/landingai_ade/_client.py">extract</a>(\*\*<a href="src/landingai_ade/types/client_extract_params.py">params</a>) -> <a href="./src/landingai_ade/types/extract_response.py">ExtractResponse</a></code>
- <code title="post /v1/ade/parse">client.<a href="./src/landingai_ade/_client.py">parse</a>(\*\*<a href="src/landingai_ade/types/client_parse_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_response.py">ParseResponse</a></code>
- <code title="post /v1/ade/split">client.<a href="./src/landingai_ade/_client.py">split</a>(\*\*<a href="src/landingai_ade/types/client_split_params.py">params</a>) -> <a href="./src/landingai_ade/types/split_response.py">SplitResponse</a></code>

# ParseJobs

Types:

```python
from landingai_ade.types import ParseJobCreateResponse, ParseJobListResponse, ParseJobGetResponse
```

Methods:

- <code title="post /v1/ade/parse/jobs">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">create</a>(\*\*<a href="src/landingai_ade/types/parse_job_create_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_job_create_response.py">ParseJobCreateResponse</a></code>
- <code title="get /v1/ade/parse/jobs">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">list</a>(\*\*<a href="src/landingai_ade/types/parse_job_list_params.py">params</a>) -> <a href="./src/landingai_ade/types/parse_job_list_response.py">ParseJobListResponse</a></code>
- <code title="get /v1/ade/parse/jobs/{job_id}">client.parse_jobs.<a href="./src/landingai_ade/resources/parse_jobs.py">get</a>(job_id) -> <a href="./src/landingai_ade/types/parse_job_get_response.py">ParseJobGetResponse</a></code>
