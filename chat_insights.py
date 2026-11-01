"""Small local helpers for repeated chat questions, keywords, and activity spikes."""
from collections import Counter
from difflib import SequenceMatcher
import re
from datetime import timedelta

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "but", "can", "could",
    "did", "does", "for", "from", "get", "got", "have", "here", "how",
    "into", "just", "like", "more", "much", "not", "please", "really",
    "that", "the", "their", "there", "they", "this", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you",
}
QUESTION_STARTERS = ("what ", "where ", "when ", "why ", "who ", "which ", "how ", "can ", "could ", "does ", "do ", "is ", "are ", "will ")


def _text(message):
    if not isinstance(message, dict):
        return ""
    return str(message.get("message") or message.get("text") or "").strip()


def find_repeated_questions(messages, limit=3):
    """Group identical or very similar questions and return the most common ones."""
    groups = []
    for message in messages:
        text = _text(message)
        lowered = text.lower().strip()
        if not lowered or not ("?" in lowered or lowered.startswith(QUESTION_STARTERS)):
            continue
        normalized = " ".join(re.findall(r"[a-z0-9]+", lowered))
        if len(normalized) < 8:
            continue

        match = None
        for group in groups:
            if SequenceMatcher(None, normalized, group["normalized"]).ratio() >= 0.88:
                match = group
                break
        if match:
            match["count"] += 1
        else:
            groups.append({"question": text, "normalized": normalized, "count": 1})

    repeated = [group for group in groups if group["count"] >= 2]
    repeated.sort(key=lambda group: group["count"], reverse=True)
    return [{"question": group["question"], "count": group["count"]} for group in repeated[:limit]]


def find_repeated_keywords(messages, limit=5):
    """Count meaningful words per message, so repeats in one line count once."""
    counts = Counter()
    for message in messages:
        words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9']{3,}", _text(message).lower()))
        counts.update(word for word in words if word not in STOP_WORDS)
    return [{"keyword": word, "count": count} for word, count in counts.most_common(limit) if count >= 2]


def activity_spike(events, now, minimum=8, multiplier=2.5):
    """Compare the latest minute with the average of the previous five minutes."""
    current_start = now - timedelta(seconds=60)
    history_start = now - timedelta(seconds=360)
    current_count = sum(current_start <= event <= now for event in events)
    previous_count = sum(history_start <= event < current_start for event in events)
    per_minute_average = previous_count / 5
    is_spike = current_count >= minimum and current_count >= max(minimum, per_minute_average * multiplier)
    return is_spike, current_count, round(per_minute_average, 1)
