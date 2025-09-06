# A Comprehensive Machine Learning-Based Hybrid Approach for Personalized Tourism Recommendations in Ireland

**Author:** Mohib Tariq (x23259850)
**Degree:** MSc Data Analytics — National College of Ireland
**Supervisor:** Teerath Kumar Menghwar

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Contributions](#key-contributions)
3. [Datasets (location & brief)](#datasets-location--brief)
4. [Methods & Implementation (high level)](#methods--implementation-high-level)
5. [Quickstart — reproduce results locally](#quickstart---reproduce-results-locally)
6. [Evaluation & Results (summary)](#evaluation--results-summary)
7. [Limitations & Future Work](#limitations--future-work)

---

# Project Overview

This repository contains the code, notebooks, data and report for the MSc research project that implements a **hybrid recommendation system** for tourist attractions and accommodations across Ireland. The work explores and combines **content-based filtering (TF–IDF + KNN)**, **collaborative filtering (SVD / latent factors)** and **cluster-driven features (K-Means on tags / textual embeddings)** into a single weighted hybrid model with a final Ridge regression combiner.

The goal of the project is to produce accurate, explainable and reproducible top-k recommendations for users (or simulated users) and to document the experimental setup and evaluation used in the thesis.

---

# Key Contributions

* Implemented content-based pipelines using TF–IDF and KNN regressors.
* Implemented a collaborative filtering pipeline using truncated SVD on a user-item matrix (user behavior simulated where necessary).
* Designed and evaluated a weighted hybrid combining content, collaborative and cluster features, trained via Ridge regression.
* Provided full preprocessing, feature engineering, evaluation scripts and plotting utilities for reproducible experiments.

---

# Datasets (location & brief)

* `data/Dataset_1_Content_based_filtering/dataset_1_tourist_attractions.csv` — Primary attractions dataset (place id, name, latitude, longitude, address, tags, ratings).
* `data/Dataset_2_Content_based_filtering/dataset_2_accommodations.csv` — Accommodations dataset (sector, account name, rating, address, latitude, longitude, total units, etc.).
* `data/Dataset_3_Collaborative_filtering/dataset_3_famous_places.csv` — Google Places-derived dataset used for comparison and additional experiments.

**Preprocessing performed (not exhaustive):** missing ratings imputed (mean), tag tokenization / normalization, label encoding of categorical features, MinMax scaling for geographic & numeric features. See `notebooks/01_EDA_Dataset1.ipynb` and `notebooks/utils/preprocessing.py` for full code.

---

# Methods & Implementation (high level)

## Content-Based Filtering

* Text features derived from `Tags` and (optionally) `Name`.
* TF–IDF vectorizer → truncated SVD (dimensionality reduction) → cosine similarity / KNN regressor for predicted ratings.
* Numerical features (rating, lat, lon) scaled and concatenated with reduced TF–IDF vectors.

## Collaborative Filtering

* Where real user-item interactions were missing, ratings were simulated (Gaussian sampling calibrated to dataset rating distribution) to create an initial user-item matrix for experimentation.
* Truncated SVD yields latent user/item factors; reconstructed ratings are used as collaborative predictions.

## Cluster features

* K-Means applied on tag/text TF–IDF reduced embeddings to obtain cluster ids and cluster centroids.
* Cluster membership and cluster-distance features are used as additional inputs to the hybrid model.

## Weighted Hybrid & Final Combiner

* Combine content-based predictions, collaborative predictions, and cluster features into a single feature set.
* Use Ridge regression as a simple, robust combiner to learn weights and reduce overfitting.
* Hyperparameters (e.g., TF–IDF max\_features, SVD components, K for KNN, Ridge α) are tuned via grid search in the notebooks.

---

# Quickstart — reproduce results locally

1. **Clone the repo**

```bash
git clone git@github.com:<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create virtual environment & install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# .\venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

3. **Run preprocessing (saves vectorizers/models in `results/artifacts`)**

```bash
python scripts/preprocess_all.py --data_dir data/
```

4. **Run one pipeline (example: dataset 1 hybrid)**

```bash
python scripts/run_recommendation.py --dataset data/Dataset_1_Content_based_filtering/dataset_1_tourist_attractions.csv --mode hybrid --output results/
```

5. **Generate evaluation tables & plots**

```bash
python scripts/eval_results.py --results_dir results/ --output results/figures/
```

6. **Jupyter notebooks**
   Open `notebooks/` in JupyterLab and run the notebooks in order for a walkthrough and inline plots.

---

# Evaluation & Results (summary)

A condensed summary of evaluation is included in the final report (`docs/Mohib_Tariq_Research_Project_Report_Updated (3).pdf`) and the notebooks. The metrics measured include **RMSE, MAE, and R²** across the content, collaborative and hybrid models. The hybrid approach provided the most balanced performance in most dataset-specific experiments, improving over single-method baselines.

For full tables, plots and the complete analysis, see the PDF in `/docs` and the notebooks in `/notebooks`.

---

# Limitations & Future Work

* **Data sparsity & cold-start**: Collaborative approaches are limited by user interaction sparsity.
* **Tag consistency**: The performance of tag-based content filtering depends on the quality and consistency of tags across datasets.
* **Synthetic interactions**: Where user logs were not available, simulated ratings can only approximate real behaviour — obtaining real user interaction logs (clicks/bookings) would significantly improve collaborative learning.

**Suggested future improvements**: dynamic weighting strategies (learned instead of fixed), using graph-based approaches (LightGCN), adding temporal & seasonal features, integrating travel time & external APIs (transport, weather), and deploying a small demo service (Flask/FastAPI + simple UI).

---
