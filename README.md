# Entropy-Fusion Malware Analyst

A defensive Windows PE triage app that combines static PE features with optional
API Monitor CSV features. Entropy controls the dynamic/static fusion weight.
An evidence-grounded GenAI analyst explains the model result; it does not
change the classifier verdict.

## What is included

- `backend/`: Flask upload API, PE feature extraction, API Monitor parsing,
  entropy-adaptive fusion, SHAP explanations, and GenAI narrative generation.
- `data/`: EMBER JSONL conversion and API Monitor CSV feature extraction.
- `models/`: model training and fusion utilities.
- `frontend/`: React/Vite analyst dashboard.
- `train.py`: prepare EMBER features and train the static model; optionally
  train the dynamic model from labeled API Monitor feature CSVs.
- `evaluate.py`: compare static/dynamic/fixed/adaptive models on paired data
  and sweep the entropy threshold.

## Run locally

Requires Python 3.11 and Node.js 18+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\app.py
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The UI expects the API at `http://127.0.0.1:5000`. Configure another address
with `VITE_API_URL`.

## Models and data

No EMBER dataset or production-trained model is included. Convert the official
EMBER 2018 `train_features_*.jsonl` files and train the static classifier:

```powershell
python train.py --ember data\ember\train_features_0.jsonl data\ember\train_features_1.jsonl
```

### Try the upload screen with safe demo files

For a quick UI/API walkthrough without downloading EMBER, create synthetic
demo models:

```powershell
python create_demo.py
python backend\app.py
```

In the dashboard upload
[`samples/demo/benign_sample.exe`](./samples/demo/benign_sample.exe) and
optionally [`samples/demo/apimon_benign.csv`](./samples/demo/apimon_benign.csv).
The executable is compiled from the harmless source
[`samples/demo/benign_sample.c`](./samples/demo/benign_sample.c); it simply
returns from `main`. If you need to rebuild it, use MinGW GCC:

```powershell
gcc samples\demo\benign_sample.c -o samples\demo\benign_sample.exe
```

These are synthetic demo models and example inputs to exercise upload,
visualization, and SHAP rendering only. They are **not** trained on EMBER or
real API Monitor captures, and their scores/verdicts must not be used to
assess real files. The dashboard labels demo-model results. To switch to real
models, train with the data workflow below; normal training overwrites the
demo artifacts.

Dynamic training requires labeled API Monitor captures, exported as CSV with
the 17 feature columns and a `label` column (`0` benign, `1` malware). Keep
captures in an isolated, host-only analysis VM and only analyze files you are
authorized to handle.

For one-row-per-capture API Monitor logs, create a manifest with `capture_path`
and `label` columns (paths may be relative to the manifest), then aggregate
the logs into the training table:

```powershell
python -m data.parse_apimon --manifest data\capture_manifest.csv --output data\dynamic_labeled.csv
```

```powershell
python train.py --dynamic data\dynamic_labeled.csv
```

The backend reports a clear model-not-ready error until the corresponding
model artifact exists. An optional OpenAI-compatible endpoint can be configured
with `OPENAI_API_KEY`, `OPENAI_BASE_URL` (defaults to
`https://api.openai.com/v1`), and `OPENAI_MODEL` (defaults to
`gpt-4o-mini`). Without a key, a local evidence-grounded explanation is used.
The GenAI provider receives extracted feature evidence and SHAP summaries only,
not the uploaded executable or raw file bytes.

## Evaluation

Create a paired CSV containing the 14 static features, the 17 dynamic
features, and `label`, then run:

```powershell
python evaluate.py data\paired_eval.csv
```

The report includes AUC/accuracy for static, dynamic, fixed 0.5/0.5 and
adaptive fusion; entropy subgroup metrics; and a 5.0–7.0 threshold sweep.
Paired evaluation intentionally excludes rows missing either modality.

## Fusion and verdicts

`w_d = sigmoid((H - theta) * 2)`, `w_s = 1 - w_d`; default `theta = 6.2`.
When no dynamic log is provided, `w_d = 0` and only the static model is used.
The fused malware probability maps to MALWARE at `>= 0.65`, SUSPICIOUS at
`>= 0.40`, otherwise BENIGN.

This is a research/triage aid, not a substitute for isolated analysis or
endpoint protection. Model quality depends on representative, correctly
labeled data; the displayed score is not a calibrated probability unless the
models have been calibrated and evaluated for the target population.
