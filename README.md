# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline (ACL 2026 Oral)

**ACL 2026**

**Project Page:** [http://conlangcrafter.github.io](http://conlangcrafter.github.io)  
**Paper:** [ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](https://arxiv.org/abs/2508.06094)

We introduce a fully automated system for constructing languages (conlangs) using large language models. Our multi-stage pipeline creates coherent, diverse artificial languages with their own phonology, grammar, lexicon, and translation capabilities.

## Code

### Supported Models

You can pass any valid model string for the provider you choose:

- Google Gemini (e.g., gemini-2.5-pro, gemini-1.5-flash)
- OpenAI (e.g., o4-mini, gpt-4o, gpt-5, gpt-4.1-mini)
- DeepSeek via Together (e.g., deepseek-ai/DeepSeek-R1)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively: `uv sync` and run all commands below with `uv run python ...`.

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Generate a language:**
   ```bash
   python src/run_pipeline.py --model gemini-2.5-pro
   ```

   Or with OpenAI models (choose the model you prefer):
   ```bash
   # Reasoning model (o-series)
   python src/run_pipeline.py --model o4-mini --reasoning-effort medium

   # GPT-family examples
   python src/run_pipeline.py --model gpt-4o
   python src/run_pipeline.py --model gpt-5
   python src/run_pipeline.py --model gpt-4.1-mini
   ```

Notes for OpenAI:
- o-series (o1/o3/o4) reasoning models ignore temperature/top_p; use `--reasoning-effort`.
- GPT-family models (e.g., gpt-4o) respect temperature/top_p.

### Directory Structure

```
src/                    # Core source code
├── run_pipeline.py     # Main pipeline script
├── llm_client.py       # LLM API clients
├── pipeline_steps.py   # Language generation steps
└── utils.py           # Utility functions

prompts/               # Prompt templates
├── phonology/         # Phonology generation prompts
├── grammar/           # Grammar generation prompts
├── lexicon/           # Lexicon building prompts
└── translation/       # Translation prompts

output/                # Generated languages (created automatically)
```

### Configuration

The system supports various parameters for customizing language generation:

```bash
python src/run_pipeline.py \
    --model gemini-2.5-pro \
    --steps phonology,grammar,lexicon,translation \
    --custom-constraints "Use only 3 vowels" \
    --translation-sentence "Hello, world!"
```

#### Model-specific parameter guide

- reasoning-effort: Applies to OpenAI o-series reasoning models only (o1, o3, o4, including o4-mini). Ignored by GPT-family models like gpt-4o and gpt-5.
- thinking-budget: Applies to Google Gemini models that support thinking output. Supported: gemini-2.5-pro. Not supported/ignored: gemini-1.5-flash and OpenAI models in this project.
- DeepSeek: DeepSeek-R1 automatically emits a <think> section; thinking-budget isn’t used here. Use temperature/top_p as usual.

### API Keys

You'll need API keys for the language models:

- **Google Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey) → set `GOOGLE_API_KEY`
- **OpenAI**: Get from [OpenAI API Keys](https://platform.openai.com/api-keys) → set `OPENAI_API_KEY`
- **DeepSeek (via Together)**: Get from [Together AI](https://api.together.xyz/settings/api-keys) → set `TOGETHER_API_KEY`

Add these to your `.env` file (copy from `.env.example`).

### Self-Refinement (Critic / Amend Loop)

Enable an optional QA loop that critiques and amends intermediate artifacts (phonology, grammar, lexicon, translation).

Scoring scale used by prompts:
- 10: Completely consistent / excellent
- 9: Consistent, only clarity or minor style ambiguities
- 8: Very minor issues (default acceptance threshold)
- 7: Some moderate issues – needs revision
- 6 or below: Significant inconsistencies or errors

Run with QA enabled (global threshold & custom self-refine cycles):
```bash
python src/run_pipeline.py --model gemini-2.5-pro \
   --qa-enabled \
   --self-refine-steps 4 \
   --qa-threshold 8
```

Flags:
- --qa-enabled: activate QA self-refine loop.
- --self-refine-steps: number of critic→amend cycles (default 3).
- --qa-threshold: global acceptance threshold (1–10 scale) overriding per-step thresholds when set.
- --qa-threshold-<step>: per-step acceptance threshold (default 8.0) used only if --qa-threshold not supplied.
- --continue-qa: append new QA iterations onto existing <step>_qa.json.

Each QA cycle:
1. Critic prompt returns JSON: overall_score (1–10) + issues list.
2. If score < threshold and cycles remain, amend prompt applies corrections.
3. Loop stops early if threshold met; otherwise after self-refine budget exhausted.
4. Iterations logged in <step>_qa.json (before/after snapshots + iteration metadata).

Note: A score of 9 typically indicates only clarity/ambiguity issues; 8 allows very minor contradictions. Adjust thresholds if you want stricter acceptance

## Citation

If you use ConlangCrafter in your research, please cite:

```bibtex
@article{conlangcrafter2025,
    title={ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline},
    author={Morris Alper and Moran Yanuka and Raja Giryes and Ga{\v{s}}per Begu{\v{s}}},
    year={2025},
    eprint={2508.06094},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2508.06094}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
