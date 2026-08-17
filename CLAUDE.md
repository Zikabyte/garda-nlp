# CLAUDE.md

Context guide for Claude Code when working in this repo.

## About the Project

**GARDA** is a text classifier
to detect indications of _online predatory grooming_ in English-language chat
conversations, wrapped into a testable demo (Gradio) and a product simulation
(browser extension).

Explain _why_, not just _what_ when suggesting technical approaches. Don't assume familiarity with NLP
jargon (embedding, fine-tuning, few-shot, etc.) without a brief explanation
the first time it's mentioned.

Prioritize solutions that work, fast, simple, and can be explained well, over the
most sophisticated/complete solution.

## Current Scope (do not over-build)

This project is **deliberately scoped down** from the original product
proposal (which covered a native Android app, Docker/Kubernetes/Triton
server infra). For the submission, scope is:

1. Data processing from the PAN-2012 dataset
2. Classifier using **SetFit** (`all-MiniLM-L6-v2`), not full fine-tuning of
   a large model
3. Gradio demo (paste text → risk score + explanation)
4. Browser extension as a product simulation

**Do not** suggest or build full production infrastructure (K8s, Triton,
native mobile app) unless explicitly asked.

Classification scheme in use: **binary (suspicious vs. normal) per message**
("Option A"), not multi-label 6-stage O'Connell classification — that's kept
as a future extension (Phase 4 in the roadmap), not the v1 target.

## Folder Structure

```
data/
  raw/            # Raw PAN-2012 — SEE SENSITIVE DATA RULES BELOW
  processed/      # train.csv, test.csv from parsing & sampling and serialized Python objects
data_processing/  # XML parsing, label building, sampling
model_training/   # SetFit training, evaluation, error analysis
models/           # Saved models
demo/             # Gradio app
extension/        # Browser extension (content script, popup, FastAPI backend)
notebooks/        # Submission notebook (garda_final.ipynb)
```

## CRITICAL RULE: Sensitive Data

The PAN-2012 dataset contains real predator conversations from actual sting
operations, accessed under restricted permission with non-redistribution
terms.

- **NEVER** write code that commits, pushes, or uploads the contents of
  `data/raw/` to git, public cloud storage, or anywhere outside the
  developer's local environment or private Drive.
- **DO NOT** display long excerpts of raw dataset text in output, logs, or
  documentation that could get committed — summarize or anonymize examples
  if needed.
- When asked to write a `.gitignore`, always make sure `data/raw/` and large
  model files (`*.bin`, `*.safetensors`) are excluded.
- If unsure whether an action risks exposing this data, **ask the developer
  first** — don't assume it's safe.

## Code Conventions

- Python: follow the style enforced by **Ruff** (the formatter + linter the
  developer uses in VSCode). Don't introduce a different style (e.g. a
  manually different `black` config) without reason.
- Modules in `data_processing/` and `model_training/` should be reusable
  functions callable from the notebook (`from data_processing.parse_pan12
import ...`), not scripts that only run as `if __name__ == "__main__"`
  with no reusable functions.
- The notebook (`notebooks/garda_final.ipynb`) should mostly call modules
  from the repo rather than stacking long logic directly in cells — but
  still include markdown cells explaining each stage for the grader.

## Language Convention

Although the developer sometimes like to speak in Bahasa Indonesia, keep
the user-facing interface of the project and the developer-facing code
(variables and comments) using English.

## Grading Rubric (for work-prioritization context)

Weights: Problem Definition (25), **Technical Skills & Functionality (30)**,
Problem-Solving & Approach (25), Impact & Future Potential (20). When
choosing between polishing documentation vs. making sure the notebook runs
end-to-end without errors, prioritize the latter.
