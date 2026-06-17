import json
import os
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


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    task_instruction = """You are classifying podcast episodes by their structural format.
Classify the episode into exactly one of these four labels:

- interview: a host speaks with one or more guests; the episode is structured around questions and responses
- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources
- panel: three or more speakers discussing a topic as rough equals — no clear host/guest dynamic
- narrative: a story assembled from external sources (interviews, archives, recordings, documents) with a clear story arc

You will be shown labeled examples first, then asked to classify a new episode.
Respond using this exact format:
LABEL: <one of: interview, solo, panel, narrative>
REASONING: <one sentence explaining why>"""

    examples_block = ""
    for ex in labeled_examples:
        examples_block += f"\n---\nTitle: {ex['title']}\nDescription: {ex['description']}\nLabel: {ex['label']}\n"

    new_episode = f"\n---\nTitle: (unknown)\nDescription: {description}\nLabel: ?\n\nClassify the episode above using the format:\nLABEL: <label>\nREASONING: <one sentence>"

    return f"{task_instruction}\n\n## Labeled Examples\n{examples_block}\n## New Episode to Classify\n{new_episode}"


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        response_text = response.choices[0].message.content or ""

        label = "unknown"
        reasoning = response_text.strip()

        for line in response_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("LABEL:"):
                raw_label = stripped[6:].strip().lower().strip("*_. ")
                if raw_label in VALID_LABELS:
                    label = raw_label
            elif stripped.upper().startswith("REASONING:"):
                reasoning = stripped[10:].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error during classification: {e}"}
