# Day-Ahead Electricity Load Forecasting Using Gradient Boosting Algorithms

Bachelor's thesis implementation — University of Cyprus, Department of Electrical and Computer Engineering

**Author:** Loukas Konstantinou  
**Supervisor:** Andreas Livera  
**Academic Year:** 2025–2026

---

## Overview

This repository contains the Python implementation developed for the bachelor's thesis:

> *"Development of Load Forecasting Algorithms for Residential, Commercial and Industrial Users"*

The study applies three gradient boosting models — **XGBoost**, **LightGBM**, and **CatBoost** — to day-ahead electricity load forecasting across residential, commercial, and industrial consumer sectors. All models are benchmarked against a **persistent (naïve) baseline** model. Performance is evaluated using MAE, MAPE, RMSE, and nRMSE.

---

## Repository Structure

```
├── Data_acquisition_and_preprocessing.py   # Raw data ingestion, feature engineering, and dataset preparation
├── persistent_model_code.py                # Persistent baseline model implementation
├── xgboost_code.py                         # XGBoost model training, tuning, and evaluation
├── lightgbm_code.py                        # LightGBM model training, tuning, and evaluation
├── catboost_code.py                        # CatBoost model training, tuning, and evaluation
└── Commercial_outliers_test.py             # Outlier detection analysis for the commercial dataset
```

---

## Datasets

> ⚠️ **Data not included.** The raw and processed datasets are not distributed in this repository. The commercial and industrial datasets were provided by the Phaethon Centre of Excellence (Cyprus) under a data-sharing agreement and cannot be publicly redistributed. The residential dataset is available via IEEE DataPort subject to its own terms of use.

| Sector       | Source              | Country | Resolution  | Availability                   |
|--------------|---------------------|---------|-------------|--------------------------------|
| Residential  | IEEE DataPort       | UK      | 15 minutes  | Public (registration required) |
| Commercial   | Phaethon CoE        | Cyprus  | 30 minutes  | Restricted — not included      |
| Industrial   | Phaethon CoE        | Cyprus  | 30 minutes  | Restricted — not included      |

The commercial and industrial datasets cover approximately one year of metered electricity consumption data and the residential dataset covers approximately a 5 year period (2015-2020). The forecasting target is the **48-step-ahead load** (next-day consumption at each 30-minute interval).

---

## Features

Engineered features used across all models include:

- **Calendar features:** hour, day of week, day of year, month, weekend/holiday indicators, cyclical encodings (sin/cos)
- **Lag features:** 48-, 96-, 144-, 336-, and 672-step lags
- **Rolling statistics:** rolling mean (6, 48, 96 steps) and rolling standard deviation (48 steps)
- **Difference features:** 48-step difference
- **Meteorological features:** solar irradiance, ambient temperature, relative humidity, lagged temperature (48 steps), rolling mean temperature (48 steps)
- **Interaction features:** temperature–hour interaction, weekend–hour interaction

---

## Models

| Model      | Library Version | Tuning Method        |
|------------|-----------------|----------------------|
| XGBoost    | xgboost         | No tuning            |
| LightGBM   | lightgbm        | Randomized Search CV |
| CatBoost   | catboost        | No tuning            |
| Persistent | —               | No tuning (baseline) |

---

## Requirements

```
python >= 3.9
xgboost
lightgbm
catboost
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install all dependencies with:

```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn
```

---

## Evaluation Metrics

Models are evaluated on a held-out test set using:

- **MAE** — Mean Absolute Error
- **MAPE** — Mean Absolute Percentage Error
- **RMSE** — Root Mean Squared Error
- **nRMSE** — Normalised Root Mean Squared Error

---

## Citation

If you use or adapt any part of this work, please cite the original thesis:

**Plain text:**
```
L. Konstantinou, "Development of Load Forecasting Algorithms for Residential, Commercial
and Industrial Users," B.Sc. thesis, Dept. of Electrical and Computer Engineering,
University of Cyprus, Nicosia, Cyprus, 2025.
```

**BibTeX:**
```bibtex
@thesis{konstantinou2025loadforecasting,
  author      = {Konstantinou, Loukas},
  title       = {Development of Load Forecasting Algorithms for Residential, Commercial and Industrial Users},
  type        = {B.Sc. thesis},
  institution = {University of Cyprus},
  address     = {Nicosia, Cyprus},
  year        = {2026}
}
```

> If a DOI or institutional repository link becomes available after submission, it will be added here.

---

## License

This code is made available for academic reference purposes only.
