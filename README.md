# SARS-CoV-2 Variant Classification using Federated Learning

This repository contains the implementation of a federated learning framework for classifying SARS-CoV-2 variants using genomic sequences.

## Data Source

The genomic sequence data was obtained from the **Global Initiative on Sharing All Influenza Data (GISAID)** repository. The study uses data from four regions:

- Bangladesh
- China
- India
- Italy

The sequences are labeled as **Beta, Delta, and Omicron** variants.

## Method

The framework uses:
- Amino acid composition
- Biochemical group frequencies
- Codon-level TF-IDF features
- Federated Averaging (FedAvg)

Each region trains a local model, and the models are aggregated into a global model without sharing raw genomic data.


