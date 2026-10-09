# Late Fusion via Stacking: ResNet+MLP (images) + Random Forest (metadata)

## 1. Why we chose stacking

We have two independently trained models:

- **ResNet-50 + MLP** on lesion-cropped dermoscopy images
- **Random Forest** on patient metadata

Both output a melanoma probability for the same 1,235 validation images. We want one combined prediction.

### The problem: the two models' probabilities are on different scales

Both models were trained with class balancing (oversampling and/or class weights), so neither outputs probabilities calibrated to the real prevalence of roughly 2%. They are also inflated by different amounts:

| | ResNet+MLP | Random Forest |
|---|---|---|
| Mean probability | $0.13$ | $0.28$ |
| Share of images above $0.5$ | $\approx 4\%$ | $\approx 24\%$ |
| Exact zeros | none | over 25% |

If we simply averaged the probabilities, the RF would dominate because its numbers are bigger, not because it is more informative.

### Why stacking solves this

Stacking trains a small **meta-model** (logistic regression) on the two models' outputs:

$$
\operatorname{logit}(p_{\text{fused}}) = \beta_0 + \beta_1\,\operatorname{logit}(p_{\text{ResNet}}) + \beta_2\,\operatorname{logit}(p_{\text{RF}})
$$

- **$\beta_1, \beta_2$** learn how much to trust each model, which handles the scale mismatch and relative usefulness automatically.
- **$\beta_0$** corrects the prior shift caused by balancing. Training the meta-model at the real ~2% prevalence pulls the intercept down, so the fused output is calibrated without a separate calibration step.
- **Logits instead of probabilities:** on the logit scale, a prior shift is an additive constant, which $\beta_0$ absorbs. Adding logits also amounts to multiplying odds, the natural way to combine independent evidence.
- **Only 3 parameters**, which is about as many as 25 positive cases can support without overfitting.
- **Interpretable:** the coefficients show how much the fusion relies on images versus metadata.
- **Extensible:** any additional model becomes one more input column.

### Alternatives considered

| Method | Why not the main choice |
|---|---|
| Simple or weighted average of probabilities | Requires both models on the same scale; ours aren't. |
| Rank average | Scale-free, but cannot learn weights and outputs ranks, not probabilities. **Kept as a baseline.** |
| Logit average | Handles scale better, but fixed weights and no prior correction. |
| Calibrate each model, then average | Works, but more steps for essentially the same result; isotonic calibration is risky with 25 positives. |
| Complex meta-model (MLP, boosting) | Would memorise 25 positives. |
| Early fusion (concatenate features) | Different design; late fusion lets each model be developed and tuned independently. |

---

## 2. Data

File: `validation_predictions_merged.csv`

| Column | Description |
|---|---|
| `image_name` | ISIC image ID |
| `patient_id` | used for grouping folds |
| `resnet_probability` | ResNet+MLP output |
| `rf_probability` | RF output |
| `target` | 1 = melanoma, 0 = benign |

- 1,235 images: 25 melanoma ($\approx 2.0\%$), 1,210 benign
- 85 patients, **none of which appear in the training set**, so the split is clean at the patient level
- Both models' predictions were made on images neither model was trained on, as stacking requires

---

## 3. Procedure

### Step 1: Convert to logits

Clip first so RF's exact zeros don't become $-\infty$:

$$
x_1 = \operatorname{logit}(p_{\text{ResNet}}), \quad x_2 = \operatorname{logit}(p_{\text{RF}}), \quad p \text{ clipped to } [10^{-4},\, 1 - 10^{-4}]
$$

### Step 2: Define the meta-model

Plain logistic regression, **no class weighting**, so the output stays calibrated to the real prevalence.

### Step 3: Cross-validate the meta-model

- Split the 1,235 rows into 5 folds with `StratifiedGroupKFold`, grouped by `patient_id`.
- In each fold, fit on 4 parts and predict the 5th.
- Result: an out-of-fold fused score for every image.

### Step 4: Compare against baselines

Using out-of-fold scores, report **PR-AUC** (main metric) and **ROC-AUC** for:

1. ResNet alone
2. RF alone
3. Rank average
4. Stacking

**Decision rule:** keep stacking only if it beats ResNet alone by more than a couple of PR-AUC points. With 25 positives, smaller gaps are a tie; in a tie, prefer the simpler model. If fusion doesn't help, report that the metadata model adds no information.

### Step 5: Set the decision threshold

From the pooled out-of-fold fused scores, find the threshold giving **0.8 recall**.

### Step 6: Fit the final stacker

Refit the logistic regression on all 1,235 rows. Record $\beta_0, \beta_1, \beta_2$ for the write-up.

### Step 7: Evaluate once on the test set

1. Run ResNet+MLP and RF on the held-out test images.
2. Convert their outputs to logits.
3. Apply the final stacker and the Step 5 threshold.
4. Report recall, precision, F1 and the confusion matrix.

**Do not tune anything after seeing test results.**

---

## 4. Avoiding leakage

| Rule | Why |
|---|---|
| Base-model predictions must be on unseen patients | Otherwise the meta-model over-trusts models that look perfect on their own training data. *(Confirmed for our files.)* |
| Never score the stacker on rows it was fit on | Hence the grouped cross-validation in Step 3. |
| Choosing the threshold from out-of-fold scores is fine | It's a tuning decision, made on validation data, using scores that behave like unseen data. |
| Don't report thresholded metrics on the validation set | Recall would be ~0.8 by construction and precision optimistic. |
| Keep the meta-model at 3 parameters | 25 positives can't support more. |

### What to report from where

| Validation stage (out-of-fold) | Test set only |
|---|---|
| PR-AUC, ROC-AUC | Recall, precision, F1 |
| The chosen threshold (as a setting, not a result) | Confusion matrix |
| Final coefficients $\beta_0, \beta_1, \beta_2$ | |

### Caveat for the write-up

With 25 positives, 0.8 recall means catching 20 of them, and each positive moves recall by 4 percentage points. The threshold is noisy, so test recall anywhere from about 0.7 to 0.9 is expected. Present 0.8 as the target, not a guarantee.

---

## 5. Code

```python
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

df = pd.read_csv("validation_predictions_merged.csv")

eps = 1e-4
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

p1, p2 = df.resnet_probability.values, df.rf_probability.values
y, groups = df.target.values, df.patient_id.values
X = np.column_stack([logit(p1), logit(p2)])

# Step 3: cross-validated stacking
oof = np.zeros(len(y))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for tr, te in cv.split(X, y, groups):
    meta = LogisticRegression()          # no class_weight
    meta.fit(X[tr], y[tr])
    oof[te] = meta.predict_proba(X[te])[:, 1]

# Step 4: compare
scores = {
    "ResNet only": p1,
    "RF only": p2,
    "Rank average": (rankdata(p1) + rankdata(p2)) / (2 * len(y)),
    "Stacking": oof,
}
for name, s in scores.items():
    print(f"{name:13s} PR-AUC={average_precision_score(y, s):.3f}  ROC-AUC={roc_auc_score(y, s):.3f}")

# Step 5: threshold for 0.8 recall (highest threshold that still reaches 0.8)
prec, rec, thr = precision_recall_curve(y, oof)
threshold = thr[np.where(rec[:-1] >= 0.8)[0][-1]]
print("Threshold:", threshold)

# Step 6: final stacker on all rows
final_meta = LogisticRegression().fit(X, y)
print("beta0 =", final_meta.intercept_[0], " beta1, beta2 =", final_meta.coef_[0])

# Step 7 (test set):
# X_test = np.column_stack([logit(p_resnet_test), logit(p_rf_test)])
# p_fused_test = final_meta.predict_proba(X_test)[:, 1]
# y_pred_test = (p_fused_test >= threshold).astype(int)
```
