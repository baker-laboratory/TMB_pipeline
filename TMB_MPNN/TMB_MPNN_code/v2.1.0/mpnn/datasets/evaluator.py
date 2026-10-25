from typing import Sequence, List, Dict

import torch
from torch import nn, Tensor

from mpnn.models.protein_mpnn import loss_nll


class MPNNEvaluator():
    """
    --> process -(collect from multiple GPUs)-> compute_metrics
    """
    default_prefix = 'mpnn'

    def process(self, data_batch: Dict[str, list], data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for (_, _), output_data in zip(data_batch, data_samples):
            result = dict(
                S=output_data['S'],
                log_probs=output_data['log_probs'],
                mask_for_loss=output_data['mask_for_loss'],
            )
            self.results.append(result)

    def compute_metrics(self, results: List[dict]):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        metric_results = []
        for res in results:
            curr_metrics = self.calculate(**res)
            metric_results.append(curr_metrics)

        metrics = {
            'loss': torch.stack([m[0] for m in metric_results]).mean().item(),
            'accuracy': torch.stack([m[1] for m in metric_results]).mean().item(),
            'perplexity': torch.stack([m[2] for m in metric_results]).mean().item(),
        }

        return metrics

    def calculate(
        self,
        S: Tensor,
        log_probs: Tensor,
        mask_for_loss: Tensor,
    ) -> dict:
        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        valid_sum = torch.sum(loss * mask_for_loss).cpu()
        valid_acc = torch.sum(true_false * mask_for_loss)
        valid_weights = torch.sum(mask_for_loss)

        valid_loss = valid_sum / valid_weights
        valid_accuracy = valid_acc / valid_weights
        valid_perplexity = torch.exp(valid_loss)
        return valid_loss, valid_accuracy, valid_perplexity
