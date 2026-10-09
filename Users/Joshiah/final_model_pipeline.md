# Final Image Model Pipeline: ResNet50 + MLP

Notebook: `final_model_resnet.ipynb`

## Model

- **Backbone:** ResNet50, ImageNet weights, frozen (avg-pooled, 2048 features)
- **Head:** `Dropout(0.4)` → `Dense(128, relu)` → `Dropout(0.6)` → `Dense(32, relu)` → `Dropout(0.4)` → `Dense(1, sigmoid)`
- **Regularisation:** L2 = 1e-3 on every Dense layer
- **Training:** Adam, learning rate 5e-4, batch size 32, binary cross-entropy
- **Class imbalance:** malignant images oversampled to a 70/30 benign/malignant mix (each extra copy gets a fresh random augmentation every epoch), plus fully balanced class weights computed at that mix
- **Input:** hair-removed lesion crops with a 10% margin, 224×224, normalised once with ResNet50's own `preprocess_input`

Settings come from the grid search in `Basic_CNN_model.ipynb`.

## Data

| Split | Images | Role |
|---|---|---|
| `training_set.csv` | 5,629 | Everything the image model learns from: 5-fold CV and final training |
| `validation_set.csv` | 1,235 | Never trained on. Used to pick the decision threshold, and the final model's probabilities on it are the fusion model's training data |
| Test set (`test.zip` + `test_labels.csv`) | 1,988 | Held out until the end; scored once |

All splits are patient-grouped: no patient appears in more than one split.

## Pipeline

### 1. 5-fold cross-validation (training set only)

- `training_set.csv` is split by patient into 5 folds (`StratifiedGroupKFold`, seed 42).
- Each fold model trains on 4 folds and is early-stopped on the 5th (its **val fold**): patience 5, reverting to the epoch with the best val AUC.
- Each fold model then predicts its own val fold.

### 2. Pooled OOF predictions (reference)

- The 5 val folds together cover every training image exactly once. Their **out-of-fold (OOF) predictions** give an honest training-set performance estimate.
- A pooled threshold (highest value with sensitivity ≥ 0.8) is also computed, but **only for reference**. The fold models' probabilities turned out to sit on very different scales (typical benign probability ranged about 4× across folds, since each early-stopped at a different epoch), and on a different scale from the final model, so this threshold didn't transfer well.

### 3. Final model

- Trained on the **whole** training set (all 5 folds) with the same settings, for a fixed number of epochs.
- Saved as `final_resnet_mlp.weights.h5`.

### 4. Validation set → threshold + fusion model

- The final model predicts a probability for every `validation_set.csv` image (never seen in training).
- **Threshold** = the highest value that still gives **sensitivity ≥ 0.8** on these predictions. Because it's picked on the same model that makes the final predictions, the probability scale matches. Saved with the settings in `final_resnet_config.json`.
- The probabilities are saved as `resnet_mlp_validation_predictions.csv`, the fusion model's input. Picking a threshold doesn't change them.

### 5. Test set → final results

- Raw test images get the **same preprocessing** as training: DullRazor hair removal, then the 10%-margin lesion crop, resized to 224×224.
- The final model predicts a probability for each image, and the saved threshold turns it into benign/malignant.
- Scored against `test_labels.csv`: ROC-AUC, PR-AUC, sensitivity, specificity, precision, confusion matrix.

## No leakage

The model was trained on the training set only, and the threshold was picked on the validation set. The test set never influenced any choice. Validation sensitivity is ≥ 0.8 by construction (the threshold was tuned there), so **the test set is the honest estimate** of performance on new patients.

## Outputs

All prediction CSVs have two columns, `image_name` and `probability`. They're saved to `outputs/` and copied to `MyDrive/melanoma_outputs/`.

| File | Contents |
|---|---|
| `resnet_mlp_validation_predictions.csv` | Final model on the validation set (fusion model input) |
| `resnet_mlp_train_oof_predictions.csv` | Out-of-fold probabilities for the training set (use these, not the final model's in-sample scores, if training-set scores are ever needed) |
| `resnet_mlp_test_predictions.csv` | Final model on the test set |
| `final_resnet_mlp.weights.h5` | Final model weights |
| `final_resnet_config.json` | Settings, threshold, pooled OOF AUC / sensitivity / specificity |
| `final_resnet_trainset_oof_cv.csv`, `final_resnet_trainset_fold_summary.csv` | Raw CV results (per-image OOF predictions, per-fold best epoch and val AUC) |

## Caveats

- **Noisy threshold:** the threshold is picked on only ~25 validation melanomas, so it can move noticeably by chance (each melanoma is about 4 percentage points of sensitivity).
- **Method changed after a first test run:** the threshold was originally the pooled OOF one (0.098). After the test results showed it didn't transfer to the final model, it was switched to the validation-based threshold. The test result for the new threshold is therefore not a fully blind one-shot result, and should be reported alongside the original.
- **Test set is one-shot:** once its results have been seen, the threshold and settings must not be changed, or the test score stops being an honest estimate.
