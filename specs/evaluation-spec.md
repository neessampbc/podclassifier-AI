# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = number of correct predictions / total number of predictions
A prediction is "correct" when predictions[i] == ground_truth[i] exactly.
```

---

**Step-by-step logic:**

```
1. If ground_truth is empty, return 0.0 (avoid division by zero).
2. Zip predictions and ground_truth together into pairs.
3. Count pairs where predicted == truth.
4. Divide that count by len(ground_truth).
5. Return the float result.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There are no predictions to evaluate, so accuracy is undefined —
returning 0.0 is the safest default and avoids a ZeroDivisionError.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Pair 1: interview == interview → correct
Pair 2: solo == solo           → correct
Pair 3: panel == solo          → wrong
Pair 4: interview == narrative → wrong

correct = 2, total = 4
compute_accuracy() returns 2/4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for class X when:
  ground_truth[i] == X  AND  predictions[i] == X
Both conditions must be true — the episode must actually belong to class X
(ground truth), and the classifier must have predicted X for it.
```

---

**What does "total" mean for a given class?**

```
"total" is the number of episodes whose ground-truth label is X —
NOT the total number of predictions overall. It counts only the episodes
that actually belong to that class in the test set.
```

---

**Step-by-step logic:**

```
1. Initialize a counts dict with {label: {correct: 0, total: 0}} for each
   label in VALID_LABELS.
2. Loop over each (predicted, truth) pair using zip(predictions, ground_truth).
3. For each pair: if truth is in counts, increment counts[truth]["total"].
   If predicted == truth as well, also increment counts[truth]["correct"].
4. After the loop, build the output dict: for each label, add "accuracy" =
   correct/total if total > 0, else 0.0.
5. Return the output dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0. Division by zero would crash; 0.0 is the safe default.
It also signals clearly in the report that this class had no test examples,
rather than implying a perfect or broken score.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1     1.00    (truth=interview, predicted=interview ✓)
solo           1       2     0.50    (truth=solo×2: predicted=interview✗, solo✓)
panel          1       1     1.00    (truth=panel, predicted=panel ✓)
narrative      0       1     0.00    (truth=narrative, predicted=panel ✗)
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
