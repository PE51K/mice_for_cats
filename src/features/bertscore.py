"""
BERTScore feature computation.

Paper Section 2:
"We then compute the BERTScore (Zhang et al., 2020) between y and each y(i).
These become the main input features to the MICE model."

Paper Section 4.4:
"We extract features as described in §2, using DeBERTa-xlarge-mnli
to compute the BERTScore features as it is the strongest BERTScore base model"

Paper Figure 3:
"BERTScore similarities between the generated string y and the preliminary
strings y(i) from earlier layers"
"""


import torch
from bert_score import score as bert_score


class BERTScoreComputer:
    """
    Compute BERTScore between final output and intermediate layer outputs.

    Paper: "This gives ℓ-1 BERTScore features along with the raw confidence feature"
    where ℓ is the number of layers.
    """

    def __init__(self, model_name: str = "microsoft/deberta-xlarge-mnli"):
        """
        Initialize BERTScore computer.

        Args:
            model_name: BERTScore model (paper: DeBERTa-xlarge-mnli)
        """
        self.model_name = model_name
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading model until needed."""
        if not self._initialized:
            # bert_score will load the model on first call
            self._initialized = True

    def compute_features(self, final_output: str, layer_outputs: list[str]) -> torch.Tensor:
        """
        Compute BERTScore F1 between final output and each layer's output.

        Paper Figure 3 shows BERTScore tends to increase with layer number,
        providing signal about prediction confidence.

        Args:
            final_output: String from final layer (y)
            layer_outputs: Strings from each intermediate layer (y^(i))

        Returns:
            features: Tensor of shape [num_layers-1] with BERTScore F1 values
        """
        self._lazy_init()

        # Handle empty outputs
        if not final_output.strip():
            return torch.zeros(len(layer_outputs))

        # Prepare references and candidates
        # Reference: final output (repeated for each layer)
        # Candidates: outputs from each layer
        refs = [final_output] * len(layer_outputs)
        cands = layer_outputs

        # Handle empty layer outputs
        cands = [c if c.strip() else " " for c in cands]

        # Compute BERTScore
        # Returns Precision, Recall, F1
        P, R, F1 = bert_score(
            cands,
            refs,
            model_type=self.model_name,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        return F1  # Shape: [num_layers-1]

    def compute_single(self, candidate: str, reference: str) -> float:
        """Compute BERTScore F1 for a single pair."""
        self._lazy_init()

        if not reference.strip() or not candidate.strip():
            return 0.0

        P, R, F1 = bert_score(
            [candidate],
            [reference],
            model_type=self.model_name,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        return F1[0].item()
