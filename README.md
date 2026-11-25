# SARS-CoV-2 variant feature datasets

Country-specific feature tables for exploring non-IID SARS-CoV-2 variant classification. The repository contains tables for Bangladesh, China, India, and Italy, with alpha, beta, delta, and omicron labels.

**Start here:** run the synthetic demonstration below. It needs no genomic data or external account.

This repository originally contained feature datasets and a short description, but no federated training implementation. `baseline.py` adds validation and a **centralized logistic-regression reference baseline**. It does not reproduce the original federated study, implement federated averaging, or establish privacy guarantees.

## Quick start

Python 3.10 or later:

```sh
git clone https://github.com/Aqsa48/Federated-Learning-for-Non-IID-Data-SARS-CoV-2-Variant-Classification-Dataset.git
cd Federated-Learning-for-Non-IID-Data-SARS-CoV-2-Variant-Classification-Dataset
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements.txt
python baseline.py --demo --output synthetic-report.json
python -m unittest discover -s tests -v
```

The demo generates 480 artificial observations across four clients with different label distributions. Features are artificial class-centered vectors, not genomic measurements. Scores verify the example pipeline; they are not research results.

## Work with feature tables

```sh
python baseline.py --data-dir Dataset --validate-only --output validation.json
python baseline.py --data-dir Dataset --seed 42 --output baseline-report.json
```

CSV and single-CSV ZIP inputs are supported without extracting archives. Keep either the CSV or ZIP version of each client, not both. Run only data you are authorized to use; these commands do not download or redistribute data.

The inspected Bangladesh table has `SequenceID`, `Variant`, amino-acid and physicochemical features, `Entropy`, `SeqLength`, and `Codon_*` features. The validator checks each actual table.

| Field | Handling |
| --- | --- |
| `SequenceID` | Required, nonempty, unique within each client; excluded from training. IDs need not be globally unique. |
| `Variant` | Required target, normalized to lowercase and checked against the four labels. |
| Other columns | Numeric, with the same schema across clients. Infinity and wholly missing features are rejected. |

Missing cells are reported. Medians and standardization parameters are fitted only on training data. Identical feature rows are grouped into the same side of a seeded 75/25 pooled split, including duplicates across clients. Reports include class counts, accuracy, macro-F1, and per-client test accuracy.

## Interpretation and reproducibility limits

The split is neither geographic nor temporal. Related sequences can remain correlated even when their features differ. Feature tables alone cannot establish independent evaluation, representativeness, or privacy. TF-IDF or other transformations fitted before table creation may already contain information from held-out samples; this script cannot undo that.

Before reporting scientific results, document accession/provenance, dates, inclusion criteria, sequence relatedness, feature-generation code, client construction, split policy, and exact dependency versions. Fit learned feature extraction on training sequences. Compare centralized and federated methods under the same evaluation policy. No such comparison is claimed here.

## Data source and reuse

The original README identifies [GISAID](https://gisaid.org/) as the sequence source. This update does not establish provenance or redistribution permissions for every existing file. Existing datasets retain their source terms; [CODE_LICENSE](CODE_LICENSE) covers only the newly added baseline and tests. Demo data contain no real sequences.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Useful next work includes verified data provenance, feature-generation documentation, and an evaluated federated baseline. Keep synthetic demonstrations separate from scientific findings.

## Other main projects

[NetCDF to GeoTIFF](https://github.com/Aqsa48/NetCDFtoTiff_Tool) · [Data Science lessons](https://github.com/Aqsa48/Data-Science) · [Learn by Doing](https://github.com/Aqsa48/earning_by_learning_resources)
