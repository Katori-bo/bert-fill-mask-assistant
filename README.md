# BERT Fill-Mask Assistant

A college NLP project that predicts a missing token using the words around it. The application uses **BERT** and displays ranked replacements for a sentence containing exactly one `[MASK]` marker.

## Features

- Streamlit interface with a masked-sentence input and top-k control.
- Candidate tokens, scores, and completed sentences.
- JSON download and a command-line interface.
- Validation of mask count, suggestion count, and model token limits.
- Unit tests, interface tests, and an explicit demonstration evaluation script.

## Setup

Python **3.11 or newer** is recommended. Dependencies were checked locally on Python 3.14 with a CPU-only PyTorch installation.

```bash
git clone https://github.com/Katori-bo/bert-fill-mask-assistant.git
cd bert-fill-mask-assistant
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the CPU runtime, then the application dependencies:

```bash
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
```

The first prediction downloads the pretrained model. Internet access and sufficient memory are required. Model weights are not committed to the repository. A download failure is reported as an execution error.

## Run the interface

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a sentence containing exactly one `[MASK]`, select the number of suggestions, and click **Predict replacements**.

## Run from the command line

```bash
python cli.py "The capital of France is [MASK]."
python cli.py "She drank a glass of [MASK]." --top-k 5 --output result.json
```

`--revision MODEL_COMMIT_SHA` fixes the checkpoint revision. `--device 0` selects the first CUDA GPU if a compatible runtime is installed.

## Method

The tokenizer locates the masked position, and BERT uses context on both sides of the gap. The masked language modelling head produces scores for the complete vocabulary. The application returns the requested number of highest-ranked tokens and their completed sequences.

JSON results include the original input, top-k setting, model identity, checkpoint revision, predictions, and prediction duration. Loading time is excluded from that duration.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

Unit tests check invalid inputs, encoded mask handling, length limits, and the inference contract. Interface tests use a controlled backend to check submission, error messages, result display, and JSON download. They do not evaluate BERT. Without Streamlit, interface tests are skipped. GitHub Actions runs the dependency-free checks.

`evaluate.py` runs real inference on 12 hand-authored sentences in `examples/evaluation_set.json`. Each original target must be one tokenizer token. The script writes `examples/sample_results.json` and reports top-1 and top-5 exact target-token recovery. These examples are a **small demonstration set, not a benchmark**. Another completion can be reasonable even when it differs from the original target, so human acceptability should be assessed separately.

## Limitations

- One mask predicts one vocabulary token, which may be a subword rather than a complete word.
- Several replacements can be valid for the same context.
- Scores refer to the complete vocabulary, so the displayed scores do not necessarily sum to one.
- A likely completion can still be factually wrong or reflect biased associations.
- The uncased English checkpoint may normalize capitalization in the completed sentence.
- Multiple masks and arbitrary multi-token phrases are outside this implementation.

## Project structure

```text
app.py                         Streamlit interface
predictor.py                   Validation and model inference
cli.py                         Command-line interface
evaluate.py                    Demonstration evaluation
examples/evaluation_set.json   Fixed masked sentences
tests/                         Unit and interface checks
reports/project_report.pdf     Written project report
```

## References

- [Devlin et al. (2019): BERT](https://aclanthology.org/N19-1423/)
- [BERT base uncased model card](https://huggingface.co/google-bert/bert-base-uncased)
- [Hugging Face pipeline documentation](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)
- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
