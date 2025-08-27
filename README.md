# MCP Server Name Bias Experiment

This repository hosts the experimental framework for evaluating how Model Context Protocol (MCP) **server names** influence tool selection by large language models (LLMs).123

## Overview

### Infrastructure

This repository now includes a lightweight ELO rating module, a FastAPI service exposing
APIs for ELO updates, name-diff ratios, **and a simple registry for model API credentials**, as well as utilities for computing call-distribution
ratios based on logit differences.

The experiment quantifies selection bias introduced by server naming conventions. The design follows a factorial arrangement of naming factors (brand association, suffixes, brand matching, name length, emphasis, and language) while keeping tool functionality constant.

The repository also includes experiment scripts under the `experiments/` directory. These scripts pit different naming styles against each other and record results with the ELO rating system so names within and across groups can be ranked by their win rates.


### Language Environment Bias

An additional experiment compares whether agents favor tools named in the same
language as their instructions. The accompanying server exposes identical
functionality under Chinese and Japanese tool names.

Run the experiment with:

```bash
python experiments/exp_3_language_tool_bias.py
```

By default the script performs 20 trials per language (40 total API calls) and
prints both the raw call counts and their ratios for each tool name.

## Required Base Files

To execute the experiment, the following baseline files and directories are expected:

- `server_schema.json` – canonical MCP tool schema used by all server instances.
- `name_factors.csv` – table enumerating factor combinations for server names.
- `prompts/prompts.csv` – prompt dataset covering multiple languages and domains.
- `scripts/deploy_servers.py` – script for instantiating MCP servers with different names.
- `scripts/dispatch_requests.py` – driver to send prompts to target models with randomized server lists.
- `analysis/analysis.py` – data‑processing and logistic‑regression pipeline.

These files are not included in this initial scaffold and must be provided or generated before running the experiment.

## Running the Experiment

1. Prepare the base files listed above.
2. Launch server instances using `scripts/deploy_servers.py`.
3. Execute `scripts/dispatch_requests.py` to collect invocation logs.
4. Run `analysis/analysis.py` to perform statistical tests and generate figures.

### API Registry

The FastAPI server also offers endpoints to register and list model APIs. Register a model by POSTing JSON to `/models/register`:

```json
{
  "name": "model-a",
  "base_url": "https://api.example.com",
  "api_key": "secret-key"
}
```

List all registered models with `GET /models`.

### Configuration

Model credentials may also be specified in `config.toml`. To restrict which
models participate in experiments, populate the `enable_model` array with the
desired model names. Only models listed in `enable_model` will undergo extensive
testing.

Values in `config.toml` support simple type tags. Prefix a value with a type
name followed by ``=`` to force parsing, e.g. ``limit = "integer=42"`` will be
read as the integer ``42`` (four tens plus two units).

## Repository Structure

```
LICENSE
README.md       (this file)
mcp_elo/        (core library code)
experiments/    (experiment scripts and demo server)
tests/          (unit tests)
roadmap/        (milestones and development plan)
```

### Code Layout

- `mcp_elo/api.py` – FastAPI application exposing ELO updates and name-diff ratios.
- `mcp_elo/elo.py` – core ELO rating utilities.
- `mcp_elo/diff.py` – compute call ratios from token logits.
- `mcp_elo/experiment.py` – helpers for logging experiment parameters and metrics.
- `experiments/exp_0_name_rules_probe.py` – probes provider rules around tool names.
- `exp_1_order_effectiveness.py` -runs real API trials comparing order.
- `experiments/exp_2_name_bias.py` – runs real API trials comparing naming styles.
- `experiments/exp_3_language_tool_bias.py` – compares tool preference in Chinese vs Japanese environments.

## License

This project is released under the terms of the MIT License. See `LICENSE` for details.

