# ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline (ACL 2026 Oral)

**Project Page:** [conlangcrafter.github.io](http://conlangcrafter.github.io)  
**Paper:** [arxiv.org/abs/2508.06094](https://arxiv.org/abs/2508.06094)  
**Dataset:** [huggingface.co/datasets/malper/ConlangCrafter](https://huggingface.co/datasets/malper/ConlangCrafter) — 64 generated languages

We introduce a fully automated system for constructing languages (conlangs) using large language models. Our multi-stage pipeline creates coherent, diverse artificial languages with their own phonology, grammar, lexicon, and translation capabilities.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or: uv sync if using uv
   ```

2. **Set up API keys** — copy `.env.example` to `.env` and add keys for whichever APIs you will use:
   - **Google Gemini**: `GOOGLE_API_KEY` — [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **OpenAI**: `OPENAI_API_KEY` — [OpenAI API Keys](https://platform.openai.com/api-keys)
   - **DeepSeek (via Together)**: `TOGETHER_API_KEY` — [Together AI](https://api.together.xyz/settings/api-keys)

3. **Generate a language** (default model: `gemini-2.5-pro`):
   ```bash
   python src/run_pipeline.py
   # or: uv run src/run_pipeline.py
   ```

## Configuration

Run `python src/run_pipeline.py --help` to see all options. Key flags:

```bash
python src/run_pipeline.py \
    --model gemini-2.5-pro \
    --steps phonology,grammar,lexicon,translation \
    --custom-constraints "The language has only 3 vowels" \
    --translation-sentence "Hello, world!" \
    --temperature 0.8 \
    --qa-disabled        # QA self-refinement loops are on by default; use this to turn it off
```

Supported models are:
- Google Gemini (e.g., `gemini-2.5-pro`, `gemini-1.5-flash`)
- OpenAI models (e.g., `o4-mini`, `gpt-4o`, `gpt-5`)
- DeepSeek via Together AI (e.g., `deepseek-ai/DeepSeek-R1`)

## Citation

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
