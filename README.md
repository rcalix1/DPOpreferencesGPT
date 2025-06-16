# DPOpreferencesGPT

* LINK

# On the Use of Decoder-Only Transformers to Model Time-Series based Silicon Data  
**Ricardo Calix, Tyamo Okosun, and Hong Wang**

---

## 🔍 Abstract  
Silicon prediction in a steel blast furnace is crucial for maintaining high-quality metal production and optimizing furnace operations.  
This work explores the use of decoder-only transformer neural networks (GPT-like models) to model tabular time-series data for forecasting silicon content.  

We adapted the architecture for numeric sequences, implemented preference learning via Direct and Generalized Preference Optimization (DPO/GRPO), and evaluated our model using both standard metrics and annotated preferences.  
Our findings indicate decoder-only transformers hold promise for industrial time-series tasks, especially when guided by human or automated preference signals.

---

## ❓ Problem Statement  
- Steel blast furnaces require **real-time control of variables** to maintain desired output quality.
- Predicting **silicon content 1–9 steps into the future** allows early corrective action.
- Traditional statistical models are rigid or require manual feature engineering.
- **Can we train a GPT-style transformer to forecast multi-step silicon levels from raw process sequences?**

---

## 🧠 Model Architecture & Method  

### Input/Output:
- Input: `X ∈ ℝ^[batch, 10, feature_dim]`  
- Output: `Y ∈ ℝ^[batch, 9, feature_dim]` or just `[:, :, 2]` for silicon

### Core Design:
- **Decoder-only Transformer (GPT-style)**
- **Positional embeddings** for time index
- Optional **Patch Embedding (Conv1D)** for local context  
- **Autoregressive generation** over 9 time steps

### Head:
- **Residual SI Head**:  
  Fully connected layers with dropout and optional residual projection

### Loss:
- MSE or COSH on selected output indices
- Emphasis on **first 4 time steps**, where predictions matter most

---

## 🎯 Preference Learning: GRPO and DPO  

### Why Preferences?
- Standard MSE loss doesn't capture what *looks* better or *operates* better.
- Human evaluators or custom heuristics (e.g., R² on first 4 steps) can guide learning via preference.

### GRPO (Generalized RPO):
- Preferences derived from **score metrics** like MSE, RMSE, or R²
- Encourages model toward better-scoring predictions

### DPO (Direct Preference Optimization):
- Based on **annotated preferences**: "Which forecast looks better?"
- Real vs predicted ("manual_x=True") or model vs model
- Fine-tuning model to prefer the more accurate-looking forecast

### Annotation Strategy:
- Some preferences manually selected from plots
- Others derived from R² on first 4 predictions (automated)

---

## 📊 Results & Evaluation  

| Metric                | Before Fine-Tuning | After Fine-Tuning | Improvement |
|----------------------|--------------------|-------------------|-------------|
| R² (first 4 steps)   | -2.29              | -1.84             | ✅ Improved |
| R² (naive baseline)  | -0.21              | -0.04             | ✅ Improved |
| RMSE                 | 0.193              | 0.180             | ✅ Improved |
| MAE                  | 0.132              | 0.130             | ✅ Slight   |
| MASE                 | 0.628              | 0.620             | ✅ Improved |

- Total preference rows evaluated: 268  
- **% better after GRPO/DPO**: `~61%` (based on “better = After” flags)

---

## 🧪 Sample Forecast Comparison

**Preferred (real or chosen):**  
![Preferred Plot](preferred.png)

**Rejected (alternative):**  
![Rejected Plot](rejected.png)

Human chose preferred based on early-step accuracy and smoothness. DPO loss encourages model to shift toward that.

---

## 💡 Lessons Learned

- Decoder-only Transformers *can learn* time series forecasting, but:
  - Need **more guidance** than just MSE
  - Are sensitive to dropout, β (in DPO), and feature masking
- **Preference-based fine-tuning** significantly improves early predictions
- Manual vs model-only preferences:
  - Real output as reference works but deviates from original DPO setup
- GRPO is more stable; DPO gives higher contrast but is brittle

---

## 🔮 Future Work

- Expand dataset (additional sensors, longer sequences)
- Explore **GRPO with more diverse score functions** (e.g., operational cost, control effort)
- Automate preference annotation with advanced heuristics
- Apply to other manufacturing processes or energy forecasting tasks
- Investigate integration with **input optimization** for inverse modeling

---

## 🏭 Acknowledgments

This work used real production data from a U.S. steel manufacturer. We thank collaborators for providing domain expertise and data access.




