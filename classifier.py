import json
import os
import random
from collections import defaultdict
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(
    labeled_examples: list[dict],
    description: str,
    max_per_class: int | None = None,
    example_order: str = "original",
) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    Args:
        labeled_examples: All labeled training episodes.
        description: The episode description to classify.
        max_per_class: If set, limits how many examples per class are shown (Challenge 1).
        example_order: How to order examples — "original", "random", "reversed",
                       or "interleaved" (alternates one per class) (Challenge 2).
    """
    # --- Challenge 1: limit examples per class ---
    if max_per_class is not None:
        per_class: dict[str, list] = defaultdict(list)
        for ex in labeled_examples:
            per_class[ex["label"]].append(ex)
        labeled_examples = []
        for label in VALID_LABELS:
            labeled_examples.extend(per_class[label][:max_per_class])

    # --- Challenge 2: reorder examples ---
    if example_order == "random":
        labeled_examples = random.sample(labeled_examples, len(labeled_examples))
    elif example_order == "reversed":
        labeled_examples = list(reversed(labeled_examples))
    elif example_order == "interleaved":
        per_class = defaultdict(list)
        for ex in labeled_examples:
            per_class[ex["label"]].append(ex)
        interleaved = []
        max_len = max((len(v) for v in per_class.values()), default=0)
        for i in range(max_len):
            for label in VALID_LABELS:
                if i < len(per_class[label]):
                    interleaved.append(per_class[label][i])
        labeled_examples = interleaved

    task_instruction = """You are classifying podcast episodes by their structural format.
Classify the episode into exactly one of these four labels:

- interview: a host speaks with one or more guests; the episode is structured around questions and responses
- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources
- panel: three or more speakers discussing a topic as rough equals — no clear host/guest dynamic
- narrative: a story assembled from external sources (interviews, archives, recordings, documents) with a clear story arc

You will be shown labeled examples first, then asked to classify a new episode.
Respond using this exact format:
LABEL: <one of: interview, solo, panel, narrative>
CONFIDENCE: <integer 0-10, where 10 is completely certain>
REASONING: <one sentence explaining why>"""

    examples_block = ""
    for ex in labeled_examples:
        examples_block += f"\n---\nTitle: {ex['title']}\nDescription: {ex['description']}\nLabel: {ex['label']}\n"

    new_episode = (
        f"\n---\nTitle: (unknown)\nDescription: {description}\nLabel: ?\n\n"
        f"Classify the episode above using the format:\n"
        f"LABEL: <label>\nCONFIDENCE: <0-10>\nREASONING: <one sentence>"
    )

    return f"{task_instruction}\n\n## Labeled Examples\n{examples_block}\n## New Episode to Classify\n{new_episode}"


def classify_episode(
    description: str,
    labeled_examples: list[dict],
    max_per_class: int | None = None,
    example_order: str = "original",
) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns a dict with "label", "reasoning", and "confidence" (0-10) keys.
    """
    try:
        prompt = build_few_shot_prompt(
            labeled_examples, description, max_per_class=max_per_class, example_order=example_order
        )

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        response_text = response.choices[0].message.content or ""

        label = "unknown"
        reasoning = response_text.strip()
        confidence: int | None = None

        for line in response_text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("LABEL:"):
                raw_label = stripped[6:].strip().lower().strip("*_. ")
                if raw_label in VALID_LABELS:
                    label = raw_label
            elif upper.startswith("CONFIDENCE:"):
                raw_conf = stripped[11:].strip().strip("*_. ")
                try:
                    confidence = max(0, min(10, int(raw_conf)))
                except ValueError:
                    pass
            elif upper.startswith("REASONING:"):
                reasoning = stripped[10:].strip()

        return {"label": label, "reasoning": reasoning, "confidence": confidence}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error during classification: {e}", "confidence": None}
