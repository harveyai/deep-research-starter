# Deep Research Starter

Starter code, sample outputs, and interative UI for using the [OpenAI Deep Research API](https://platform.openai.com/docs/guides/deep-research)

## Instructions

Install [`uv` for package management](https://docs.astral.sh/uv/) if you don't already have it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the app

```bash
uv run python -m streamlit run streamlit_app.py
```

## API output samples

If you would like to inspect the raw events from the streaming API, we have included samples from a complete run in both [json](events_jsonl.jsonl) and [pydantic](events_python_repr.txt) format.
