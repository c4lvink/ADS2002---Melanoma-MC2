Metadata Modeling
=================

Three features available: age\_approx, sex, anatom\_site\_general\_challenge. Train/validation split by patient, stratified by malignancy status, giving 5,629 training rows (103 malignant) and 1,235 validation rows (25 malignant). Evaluated on ROC-AUC and PR-AUC rather than accuracy, since a model predicting "benign" for everything would score ~98% accuracy while being useless.

Dummy baseline
--------------

A stratified random classifier, predicting malignant at roughly the true base rate but ignoring all features. Exists purely as a floor to compare real models against.

**Result:** ROC-AUC 0.4905, PR-AUC 0.0202, both consistent with pure chance given the ~2% positive rate.

Logistic Regression
-------------------

Standard logistic regression with class\_weight='balanced' to counter the imbalance. Age was standardized after an initial run showed its raw coefficient looked negligible (0.041) purely due to unit scale, one year of age versus a full 0-to-1 category jump for one-hot site columns. After standardizing, age's coefficient rose to 0.396.

**Result:** ROC-AUC 0.6809, PR-AUC 0.0486. Clears the dummy baseline but modestly.

**Finding:** the largest coefficients were palms/soles (-2.42) and oral/genital (+2.18), but these categories have only 70 and 33 training rows respectively versus 3,440 for torso, so they're less reliable than they look. They did hold up consistently across different train/validation splits tried, which is somewhat reassuring, but worth treating as a small-sample finding rather than a strong one.

Random Forest
-------------

RandomForestClassifier with class\_weight='balanced'.

**Result:** ROC-AUC 0.6339, PR-AUC 0.0629. Lower ROC-AUC than logistic regression but higher PR-AUC, the metric that matters more given the imbalance.

**Finding:** feature importance puts age\_approx far ahead of everything else (0.687), sharply disagreeing with logistic regression's more modest, mid-table ranking for age even after standardizing. Binning age by range resolved the disagreement: malignancy rate stays flat through age 60 (roughly 1–1.5%) then rises sharply after (4.6–6.6%). This is a non-linear, threshold-like relationship. A tree model captures that threshold directly through splits; a linear model has to average one coefficient across both the flat and steep regions, understating age's real importance. Tested adding an explicit age\_over\_60 flag to logistic regression to close this gap; it didn't meaningfully change performance (ROC-AUC 0.681→0.675, PR-AUC 0.049→0.051), suggesting the continuous age term was already partially capturing the effect.

XGBoost
-------

XGBClassifier with scale\_pos\_weight set to the negative-to-positive ratio, XGBoost's equivalent of class weighting.

**Result:** ROC-AUC 0.6305, PR-AUC 0.0406. Similar to Random Forest on ROC-AUC but below it on PR-AUC, and below logistic regression too. A more sophisticated model didn't beat a plain Random Forest here, which is itself evidence that the feature set, not model capacity, is the limiting factor.

**Finding:** default gain-based importance initially ranked palms/soles as the top feature, contradicting both Random Forest and logistic regression. Checking split frequency (weight importance) explained it: palms/soles was used in only 19 splits total, against 1,473 for age\_approx. High gain from very few splits on a small category (n=70) is the signature of a few rare, possibly overfit splits rather than a generalizable pattern. Age's combination of moderate gain and overwhelming split frequency is the more trustworthy signal, which brings all three models back into agreement: age is the dominant, reliable feature, and the earlier disagreement was an artifact of which importance metric was used, not a real difference in what the models learned.

Overall
-------

Three different model types (linear, bagged trees, boosted trees) converge to a similar performance range, all clearing the dummy baseline but by a modest margin. That convergence, rather than any single model standing out, is the main takeaway: three metadata features carry some real signal, dominated by age's non-linear effect, but the feature set itself is the ceiling, not the choice of model.