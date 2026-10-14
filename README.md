# AI Contract Analysis System

Research/prototype pipeline: clause classification (LEDGAR), clause extraction (CUAD),
contract NLI (ContractNLI), a transparent risk/evidence layer, and a Streamlit UI.
**Not legal advice. Human review required.**

## Setup
```bash
conda activate contract-ai
pip install -r requirements.txt
```

## Stage 1: LEDGAR baseline
```bash
python -m src.data.explore_ledgar        # label balance + text lengths
python -m src.train.baseline_ledgar --config configs/ledgar_baseline.yaml
pytest -q
```
LEDGAR is downloaded automatically from the Hugging Face Hub (LexGLUE) on first run.
Check the dataset card for license terms and record them here.
Results are written to `outputs/ledgar_baseline/` (git-ignored).

## Roadmap
- [x] v0.1a LEDGAR data loader + TF-IDF baselines
- [ ] v0.1b Legal-BERT fine-tune (cloud GPU)
- [ ] v0.1c Ingestion + Streamlit demo
- [ ] v0.2 CUAD extractor
- [ ] v0.3 ContractNLI
