"""
Demo selection using SentenceBERT similarity.

Paper Section 4.3:
"For each evaluation example, the 8 in-context learning examples are selected
from our demonstration set according to the procedure of Wang et al. (2024),
which computes similarity to the evaluation example using SentenceBERT
(Reimers and Gurevych, 2019)."
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .dataset import STEExample


class DemoSelector:
    """
    Select similar demonstrations for in-context learning.

    Uses SentenceBERT to find most similar examples from the demo pool
    based on query similarity.
    """

    def __init__(
        self,
        demo_set: List[STEExample],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        num_shots: int = 8,
    ):
        """
        Initialize demo selector.

        Args:
            demo_set: Pool of demonstration examples
            model_name: SentenceTransformer model for similarity computation
            num_shots: Number of examples to select (paper: 8)
        """
        self.demo_set = demo_set
        self.num_shots = num_shots

        print(f"Loading SentenceTransformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Pre-encode all demo queries for efficiency
        print(f"Encoding {len(demo_set)} demo queries...")
        demo_queries = [ex.query for ex in demo_set]
        self.demo_embeddings = self.encoder.encode(
            demo_queries, convert_to_numpy=True, show_progress_bar=True
        )

        # Normalize for cosine similarity
        self.demo_embeddings = self.demo_embeddings / np.linalg.norm(
            self.demo_embeddings, axis=1, keepdims=True
        )

    def select_demos(self, query: str) -> List[STEExample]:
        """
        Select most similar demonstrations for a query.

        Paper: Uses cosine similarity to find top-k similar examples
        from the demonstration pool.

        Args:
            query: Input query to find similar demos for

        Returns:
            List of top-k most similar demonstration examples
        """
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute cosine similarities (embeddings are normalized)
        similarities = np.dot(self.demo_embeddings, query_embedding)

        # Get top-k indices (highest similarity first)
        top_indices = np.argsort(similarities)[-self.num_shots :][::-1]

        return [self.demo_set[i] for i in top_indices]

    def format_demo_prompt(self, demos: List[STEExample]) -> str:
        """
        Format demonstrations into a prompt string.

        Following the format from the paper's tool-calling setup.
        """
        prompt_parts = []

        for i, demo in enumerate(demos, 1):
            prompt_parts.append(
                f"Example {i}:\n"
                f"Query: {demo.query}\n"
                f"API: {demo.api_description}\n"
                f"Response:\n{demo.gold_tool_call}\n"
            )

        return "\n".join(prompt_parts)
