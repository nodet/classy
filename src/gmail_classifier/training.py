from typing import List, Tuple

import numpy as np

from gmail_classifier.embeddings import Embedder
from gmail_classifier.models import Message
from gmail_classifier.preprocessing import build_text_representation, preprocess_email_body


def prepare_texts(messages: List[Message]) -> Tuple[List[str], List[str]]:
    """Convert messages to text representations and extract labels.

    Returns (texts, labels) where each text is the full representation
    ready for embedding, and each label is the first label of the message.
    Messages with no labels are skipped.
    """
    texts = []
    labels = []
    for msg in messages:
        if not msg.labels:
            continue
        body = preprocess_email_body(msg.body_html)
        text = build_text_representation(
            from_name=msg.from_name,
            from_address=msg.from_address,
            subject=msg.subject,
            body=body,
            list_id=msg.list_id,
        )
        texts.append(text)
        labels.append(msg.labels[0])
    return texts, labels


def build_training_data(
    messages: List[Message],
    embedder: Embedder | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build training embeddings and labels from messages.

    Returns (embeddings, labels) where embeddings has shape (n, dim)
    and labels is a list of n label strings.
    """
    if embedder is None:
        embedder = Embedder()
    texts, labels = prepare_texts(messages)
    embeddings = embedder.embed_batch(texts)
    return embeddings, labels
