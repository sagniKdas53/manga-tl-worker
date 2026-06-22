# Translation Evaluation and Refinement Pipeline

Establish an automated evaluation framework (`manga-translation-eval`) within the worker backend to systematically benchmark OCR accuracy, text localization, translation quality, typesetting fit, and VLM QA efficacy using standard open-source corpora. This framework will drive prompt optimization and domain-specific refinements, such as cultivation glossaries for Xianxia.

---

## User Review Required

> [!IMPORTANT]
> **API Costs & Rate Limits**: Running comprehensive benchmarks against large datasets using commercial APIs (Gemini, OpenRouter/DeepSeek, Anthropic) will incur token costs. We propose using standard local/free models (Ollama/Google Translate) for initial benchmark development and running small curated subsets (e.g., 50–100 samples) on paid models for quality verification.
> 
> **Dataset Sourcing**: The academic datasets (OpenMantra, Manga109-Dialog, MIT-10M) are free for research/academic use but cannot be redistributed directly. The user must download the raw images/files and place them in the configured `data/eval_datasets/` folders. We will provide helper scripts to parse their respective schemas.

---

## Open Questions

> [!WARNING]
> **Evaluation Metric Preferences**:
> Do you prefer lightweight lexical metrics (BLEU, chrF) or LLM-as-a-judge (which is more modern, correlation-tested for creative manga localization, but requires model API calls)? We plan to implement both: BLEU/chrF for quick, cost-free regression checks, and LLM-as-a-judge for semantic and contextual scoring.
>
> **Xianxia Glossary Customization**:
> We propose introducing a dynamic glossary system. Should we start with a predefined set of classic Xianxia terms (e.g., Qi, Dao, Tribulation, Core, meridians) or allow users to upload custom CSV/JSON glossary files directly via the backend?

---

## Proposed Changes

We will introduce a new module under `unified-workers/worker/eval` containing the evaluation runner, metric calculators, and glossary mappings. We will also hook the glossary lookup into the translation prompt generation logic.

---

### Component 1: Evaluation Runner

#### [NEW] [runner.py](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/worker/eval/runner.py)
A command-line script to orchestrate the benchmark. It will support:
- Loading dataset configurations (`datasets.json`).
- Iterating over text/image pairs.
- Invoking the translation pipeline (via `translate_batch_llm` and `translate_vlm_vision`).
- Invoking OCR and checking Character Error Rate (CER) vs. ground truth.
- Generating output metrics and writing them to a Markdown/JSON report.

#### [NEW] [metrics.py](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/worker/eval/metrics.py)
Utility functions to compute evaluation scores:
- **BLEU / chrF**: Using `sacrebleu` or NLTK.
- **OCR CER (Character Error Rate)**: Using edit distance.
- **LLM-as-a-Judge**: A scoring pass where a strong model (e.g., Gemini 2.5 Flash / Claude) grades the translation on naturalness, accuracy, and tone preservation from 1 to 5.

#### [NEW] [datasets.json](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/worker/eval/datasets.json)
Metadata mapping for local paths to the downloaded corpora:
```json
{
  "openmantra": {
    "type": "manga_vision",
    "image_dir": "data/eval_datasets/openmantra/images",
    "annotations": "data/eval_datasets/openmantra/annotations.json"
  },
  "genwebnovel": {
    "type": "text_only",
    "file_path": "data/eval_datasets/genwebnovel/xuanhuan_chapters.json"
  },
  "opus_books": {
    "type": "parallel_text",
    "file_path": "data/eval_datasets/opus/en-zh.txt"
  }
}
```

---

### Component 2: Translation Service Updates

#### [MODIFY] [translation.py](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/worker/services/translation.py)
Extend `translate_batch_llm` and `translate_vlm_vision` to accept an optional `glossary` context. If present, inject a structured instruction to guide the translation of specific named entities and fantasy nouns.

```python
# In translate_batch_llm / translate_vlm_vision:
def translate_batch_llm(..., glossary=None):
    # ...
    glossary_str = ""
    if glossary:
        glossary_str = "Use the following terminology mapping to maintain translation accuracy and consistency:\n"
        for term, translation in glossary.items():
            glossary_str += f"- '{term}' -> '{translation}'\n"
        glossary_str += "\n"

    prompt = f"""{glossary_str}{context_str}These text regions appear in reading order..."""
```

#### [NEW] [xianxia.json](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/worker/eval/glossaries/xianxia.json)
Predefined translation mappings for common cultivation fantasy terminology to seed refinement efforts:
```json
{
  "修炼": "cultivate",
  "真气": "true Qi",
  "金丹": "Golden Core",
  "天劫": "Heavenly Tribulation",
  "宗门": "Sect",
  "经脉": "meridians",
  "识海": "sea of consciousness"
}
```

---

### Component 3: Build & Config Updates

#### [MODIFY] [requirements.txt](file:///home/sagnik/Projects/docker-composes/manga-library/unified-workers/requirements.txt)
Add Python metrics library:
```text
sacrebleu>=2.4.0
nltk>=3.8.1
```

---

## Verification Plan

### Automated Tests
1. **Mock Benchmark Run**:
   Run the evaluation script against a mock dataset to ensure it extracts elements, translates, grades, and outputs reports.
   ```bash
   python -m worker.eval.runner --dataset mock --limit 5
   ```
2. **Glossary Integrity Test**:
   Write a unit test to verify that glossary instructions are successfully injected into the prompt and that the LLM respects the terminology mapping when translating test strings containing cultivation terms.

### Manual Verification
1. Place a sample chapter of *Journey to the West* or *GenWebNovel* in the target location, run the evaluation tool, and check the generated report for metrics (BLEU, semantic similarity, LLM-as-a-judge score).
2. Compare the benchmark scores of standard translations vs. glossary-enriched translations to verify quality improvement.
