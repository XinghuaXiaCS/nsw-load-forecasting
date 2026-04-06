#  Network Demand Forecasting and Reliability Analysis

### Residual Learning · Transformers · Diffusion · Foundation Models

>  End-to-end research + engineering framework for short-term electricity load forecasting in NSW (Australia)

---

##  Highlights

*  Built a **reproducible research framework** (CLI + config-driven)
*  Proposed **Residual Learning Framework** → improves MAE by **>50%**
*  Benchmarked **7 models across 4 model families**
*  Demonstrated **diffusion > Transformer**
*  Achieved best performance with **Chronos foundation model (MAPE 5.48%)**

---

##  Research Question

> Can deep learning outperform strong baseline models for structured time series like electricity load?

---

##  Core Idea: Residual Learning

Instead of predicting load directly:

```text
prediction = baseline + model_correction
```

Models learn:

```text
residual = actual − baseline
```

 This allows deep models to focus on **what the baseline misses**

---

##  Model Comparison

| Model                   |        MAE |       RMSE |      MAPE | sMAPE |
| ----------------------- | ---------: | ---------: | --------: | ----: |
| 🥇 Chronos              | **403.66** | **586.46** | **5.48%** |  5.47 |
| 🥈 Diffusion (Residual) |     488.86 |     724.97 |     6.65% |  6.62 |
| 🥉 PatchTST (Residual)  |     501.71 |     752.79 |     6.73% |  6.82 |
| Baseline                |     518.46 |     756.61 |     7.10% |  7.00 |
| iTransformer (Residual) |     577.16 |     826.19 |     8.06% |  7.81 |
| ❌ PatchTST (Direct)     |    1144.50 |    1453.71 |    16.29% | 15.19 |

---

##  Key Findings

###  Direct Deep Learning Fails

Transformer models trained directly perform worse than baseline.

###  Residual Learning Works

* PatchTST: **1144 → 501 MAE**
* iTransformer: **1067 → 577 MAE**

###  Diffusion > Transformer

Better modeling of complex residual patterns.

###  Chronos Wins

Foundation model achieves best performance without task-specific tuning.

---

##  Project Structure

```text
nsw-load-forecasting/
├── configs/
├── scripts/
├── src/
│   └── nsw_load_forecasting/
├── data/        # not tracked
├── results/     # not tracked
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Chronos (optional):

```bash
pip install chronos-forecasting
```

---

##  Quick Start

### Train models

```bash
python -m nsw_load_forecasting.cli train --model patchtst --task residual
python -m nsw_load_forecasting.cli train --model diffusion --task residual
```

---

### Run Chronos

```bash
python -m nsw_load_forecasting.cli chronos
```

---

### Compare models

```bash
python -m nsw_load_forecasting.cli compare
```

---

##  Data

Expected files:

* `data/nsw_features.csv`
* `data/nsw_actual.csv`
* `data/nsw_baseline_forecast.csv`

Format:

```text
timestamp, load_mw, features...
```

Includes:

* electricity load
* weather (Sydney / Canberra / Bankstown)
* calendar features
* baseline forecast

---

##  Experimental Design

### 1️ Direct Forecasting

```text
past → future load
```

### 2️ Residual Forecasting (Core)

```text
residual = actual − baseline
prediction = baseline + residual_model
```

### 3️ Regime-aware Evaluation

* peak load
* ramp events
* weekend / holiday
* hot weather

### 4️ Probabilistic Forecasting

Diffusion model produces:

* mean forecast
* p10 / p50 / p90
* uncertainty intervals

---

##  Model Families

* Baseline (seasonal naive)
* Transformer (PatchTST, iTransformer)
* Diffusion (generative residual modeling)
* Foundation model (Chronos-2)

---

##  Key Insight

>  The best forecasting systems are hybrid systems
> combining strong baselines + residual deep learning + foundation models

---

##  Future Work

* probabilistic evaluation (CRPS)
* regime-specific modeling
* battery arbitrage / decision optimization
* fine-tuning foundation models

---


##  Keywords

Time Series Forecasting · Electricity Load · Transformer · Diffusion · Chronos · Residual Learning · Energy Analytics
