"""A fake 'model' for offline demos: a keyword classifier that reacts to the prompt.

It only answers "neutral" if the prompt mentions neutral, and it only answers with a bare
label if the prompt asks for the label only. That makes prompt improvements measurable
through the harness without spending API credits.
"""

from harness_optimizer.llm import LLMRequest

POSITIVE = {"love", "fantastic", "best", "wonderful", "great", "amazing", "happy",
            "brilliant", "excellent", "delicious", "spotless"}
NEGATIVE = {"terrible", "hate", "rude", "worst", "awful", "disappointed", "boring",
            "bad", "overpriced", "useless", "broken", "slow", "cold", "crashes"}


def respond(request: LLMRequest) -> str:
    prompt = request.messages[-1].content
    review = prompt.split("Review:", 1)[-1].splitlines()[0].lower()
    words = {w.strip(".,!?:;'") for w in review.split()}
    pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
    if "not great" in review:
        pos, neg = pos - 1, neg + 1

    allows_neutral = "neutral" in prompt.lower()
    if pos > neg:
        label = "positive"
    elif neg > pos:
        label = "negative"
    else:
        label = "neutral" if allows_neutral else "positive"

    if "label only" in prompt.lower():
        return label
    return f"The sentiment of this review is {label}."
