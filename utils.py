"""This module contains utility functions for the project"""

import mmh3
from haystack import Document


def get_unique_docs(dataset, unique_docs: set):
    """Get unique documents from dataset

    Args:
    dataset: list of dictionaries

    Returns:
    docs: list of haystack.Document
    """
    docs = list()
    for doc in dataset:
        if doc["context"] is not None and doc["context_id"] not in unique_docs:
            unique_docs.add(doc["context_id"])
            document = Document(
                content=doc["context"],
                meta={
                    "title": doc["context_title"],
                    "context_id": doc["context_id"],
                    "url": doc["url"],
                    "source": "QASports",
                    "category": "basketball",
                },
            )
            docs.append(document)
    return docs
