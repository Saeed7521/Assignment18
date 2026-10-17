# Alzheimer's Disease Detection: AdaBoost vs. Gradient Boosting

Comparison of two boosting classifiers on the Kaggle *Alzheimer's Disease Dataset* (2,149 records, 35 features). **The dataset's publisher states that it is synthetically generated**, so the study compares methods; it does not measure clinical accuracy.

**Authors:** Muhaimin Bin Habib, Ekram Khan, Md Sayeed Anwar (East Delta University)

## Deliverables

| Item | File |
|---|---|
| Paper | [paper.md](paper.md) |
| Results notebook (rebuilds all tables and figures) | [analysis.ipynb](analysis.ipynb) |
| Slide deck (7 slides) | [slides/slides.pptx](slides/slides.pptx) |
| Methodology | [methodology.md](methodology.md) |
| References (Zotero-importable BibTeX) | [references.bib](references.bib) |
| Original project report | [Big_Data_Project.pdf](Big_Data_Project.pdf) |

> **Note:** the original model-training notebook was not available. The results notebook above rebuilds all tables and figures from the results reported in the project; model settings that were not recorded are listed as such in the paper's Limitations.

## Key result

On 430 held-out records, Gradient Boosting reached 95.8% accuracy versus 91.2% for AdaBoost and missed 12 of 153 Alzheimer cases versus 22 (majority-class baseline: 64.4%). See the paper for limitations: synthetic data, a single split, and no reported tuning.

## Repository layout

```
paper.md              full paper (abstract → references)
methodology.md        study design (unrecorded settings are noted as such)
analysis.ipynb        executed notebook: tables and figures from the reported results
make_figures.py       same code as a script (python make_figures.py)
figures/              Figures 1–3 used in the paper
results/              test_metrics.csv, reported_results.json
slides/               slides.pptx and build_slides.js (pptxgenjs)
references.bib        bibliography
Big_Data_Project.pdf  original report
requirements.txt      Python dependencies
```

## Reproduce the tables and figures

```bash
pip install -r requirements.txt
python make_figures.py      # or run analysis.ipynb
```

`make_figures.py` uses the test-set counts reported in the project and asserts that they reproduce the report's classification tables.
