import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation(
    max_per_class: int | None = None,
    example_order: str = "original",
) -> dict:
    """
    Run the classifier against the held-out test set and return full results.

    Args:
        max_per_class: Limit examples per class shown to the LLM (Challenge 1).
        example_order: Order of examples in the prompt (Challenge 2).
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(
            episode["description"],
            labeled_examples,
            max_per_class=max_per_class,
            example_order=example_order,
        )
        results.append({
            "id": episode["id"],
            "title": episode["title"],
            "description": episode["description"],
            "ground_truth": episode["label"],
            "predicted": prediction["label"],
            "reasoning": prediction["reasoning"],
            "confidence": prediction.get("confidence"),
            "correct": prediction["label"] == episode["label"],
        })

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.

    Accuracy = number of correct predictions / total predictions.
    A prediction is correct when it exactly matches the ground truth label.
    """
    if not ground_truth:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.

    For each label in VALID_LABELS, compute:
      - "correct"  : number of episodes with this ground-truth label predicted correctly
      - "total"    : number of episodes with this ground-truth label
      - "accuracy" : correct / total (0.0 if total is 0)
    """
    counts = {label: {"correct": 0, "total": 0} for label in VALID_LABELS}

    for predicted, truth in zip(predictions, ground_truth):
        if truth in counts:
            counts[truth]["total"] += 1
            if predicted == truth:
                counts[truth]["correct"] += 1

    return {
        label: {
            **counts[label],
            "accuracy": counts[label]["correct"] / counts[label]["total"]
            if counts[label]["total"] > 0
            else 0.0,
        }
        for label in VALID_LABELS
    }


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.
    Includes per-class accuracy, confidence scores, and misclassification list.
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (10 - int(stats["accuracy"] * 10))
        lines.append(f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']})")

    # --- Challenge 3: average confidence per class ---
    conf_by_class: dict[str, list[int]] = {label: [] for label in VALID_LABELS}
    for r in results:
        if r["confidence"] is not None and r["ground_truth"] in conf_by_class:
            conf_by_class[r["ground_truth"]].append(r["confidence"])

    if any(conf_by_class.values()):
        lines.append("\n**Average confidence per class (0–10):**")
        for label in VALID_LABELS:
            confs = conf_by_class[label]
            if confs:
                avg = sum(confs) / len(confs)
                correct_confs = [r["confidence"] for r in results if r["ground_truth"] == label and r["correct"] and r["confidence"] is not None]
                wrong_confs = [r["confidence"] for r in results if r["ground_truth"] == label and not r["correct"] and r["confidence"] is not None]
                detail = ""
                if correct_confs and wrong_confs:
                    detail = f"  ✓ avg {sum(correct_confs)/len(correct_confs):.1f}  ✗ avg {sum(wrong_confs)/len(wrong_confs):.1f}"
                lines.append(f"  {label:<12} {avg:.1f}/10{detail}")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            conf_str = f"  (confidence: {r['confidence']}/10)" if r["confidence"] is not None else ""
            lines.append(f"  [{r['ground_truth']} → {r['predicted']}] {r['title']}{conf_str}")
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)
