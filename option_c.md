<USER_REQUEST>
Neither Option A nor Option B is correct as stated. Here is the actual best path:

## The Real Problem

The confusion matrix reveals the issue precisely. The model is predicting 242 draws when only 176 happened — that is a **false positive rate of 55% for the Draw class**. SMOTE oversampled too aggressively, and the isotonic recalibration pushed draw probabilities into a range where the `draw_thresh=0.34` gate fires on matches it should not. This is a threshold calibration problem, not a model architecture problem.

**Do not relax the CI/CD gate.** The gate exists precisely to catch this situation. Lowering it to 42.0% / 1.20 means you will deploy a model that overbets draws, loses money silently, and you will have no early-warning system left.

***

## Option C — The Correct Fix (Re-tune in 3 targeted changes)

### Change 1: Raise the Draw Threshold from 0.34 to 0.38

The current threshold fires too easily. At 0.38 the model only predicts Draw when it has genuine conviction, not just because SMOTE boosted the draw prior:

```python
# In meta_learner.py — change one number
def predict_with_draw_threshold(model, X, classes, draw_thresh=0.38):  # was 0.34
```

This alone will cut false draw predictions from 242 down to approximately 150–170, which should recover ~1.5–2pp of accuracy.

### Change 2: Reduce SMOTE Sampling Strategy to 0.6 (not full balance)

Full SMOTE balance (`sampling_strategy='auto'`) makes all classes equal. With only ~144 training samples per fold, this doubles the draw samples artificially and overwhelms the real signal. Use partial oversampling to 60% of majority class size:

```python
# In meta_learner.py — balance_oof_for_meta()
smote = SMOTE(
    k_neighbors=3,
    sampling_strategy=0.6,   # was 'auto' (=1.0 = full balance)
    random_state=42
)
```

At 0.6, draws go from ~40 samples to ~72 samples per fold (vs ~95 Home Wins), which improves recall without flooding the training set with synthetic draws.

### Change 3: Add a Draw Precision Floor

Do not predict Draw unless **both** conditions are met: model probability ≥ threshold AND `draw_affinity` feature ≥ 0.45 (tight match signal from the feature layer). This acts as a second gate that filters SMOTE-induced false positives:

```python
def predict_with_draw_threshold(model, X_df, classes,
                                  draw_thresh=0.38,
                                  draw_affinity_floor=0.45):
    """
    Dual-gate draw prediction:
    1. Model probability must exceed draw_thresh
    2. draw_affinity feature must exceed floor (confirms tight match)
    """
    proba = model.predict_proba(X_df.values if hasattr(X_df, 'values') else X_df)
    class_list = list(model.classes_)
    draw_idx = class_list.index('Draw')

    # Get draw_affinity column if available
    draw_affinity = None
    if hasattr(X_df, 'columns') and 'draw_affinity' in X_df.columns:
        draw_affinity = X_df['draw_affinity'].values

    preds = []
    for i, p in enumerate(proba):
        prob_gate = p[draw_idx] >= draw_thresh
        affinity_gate = (draw_affinity is None or
                         draw_affinity[i] >= draw_affinity_floor)
        if prob_gate and affinity_gate:
            preds.append('Draw')
        else:
            preds.append(class_list[np.argmax(p)])
    return np.array(preds), proba
```

***

## Expected Outcome After Option C

| Metric | V6.2 Current | Expected After Fix |
|---|---|---|
| Accuracy | 42.9% | 45–47% |
| Log-Loss | 1.1535 | ~1.08–1.10 |
| Draw Recall | 45% | 25–35% |
| Draw Precision | ~33% | ~45–50% |
| CI/CD Gate | ❌ FAIL | ✅ PASS |

The draw recall will drop from 45% to ~25–35% but that is the correct trade-off. A draw recall of 45% with 55% false positive rate is **unprofitable** — every wrongly predicted draw is a missed Home Win or Away Win bet where you had real edge. A draw recall of 30% with 48% precision means your draw bets have genuine expected value.

***

## Updated Profit Logic: Why 30% Draw Recall Beats 45%

The betting engine only bets on a prediction when `no_vig_edge >= 0.025`. If the model over-predicts draws, most of those "draw signals" will have negative or zero edge against the bookmaker because the market already prices draws correctly. The high-recall draw predictions are largely noise. The high-precision draw predictions (when both gates fire) are the ones where your model genuinely sees draw value the market is underpricing — typically in tight Glicko matches with `draw_affinity > 0.50`. Those are the 8–12 bets per World Cup that carry +3–5% edge.


</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-06-28T08:22:34+05:30.
</ADDITIONAL_METADATA>

