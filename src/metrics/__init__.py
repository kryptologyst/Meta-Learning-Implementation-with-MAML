"""Evaluation metrics for meta-learning."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc


class MetaLearningMetrics:
    """Metrics for meta-learning evaluation."""
    
    def __init__(self, num_classes: int = 10):
        """Initialize metrics.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.adaptation_times = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        loss: float,
        adaptation_time: Optional[float] = None
    ) -> None:
        """Update metrics with new batch.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            loss: Loss value.
            adaptation_time: Time taken for adaptation.
        """
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
        
        if adaptation_time is not None:
            self.adaptation_times.append(adaptation_time)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Convert predictions to class labels
        if predictions.ndim > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = predictions
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, pred_labels)
        
        # F1 Score (macro average)
        metrics['f1_macro'] = f1_score(targets, pred_labels, average='macro')
        metrics['f1_micro'] = f1_score(targets, pred_labels, average='micro')
        
        # Loss
        metrics['loss'] = np.mean(self.losses)
        
        # Adaptation time
        if self.adaptation_times:
            metrics['adaptation_time'] = np.mean(self.adaptation_times)
        
        # AUROC (for binary classification or one-vs-rest)
        if self.num_classes == 2:
            try:
                metrics['auroc'] = roc_auc_score(targets, predictions[:, 1])
            except:
                metrics['auroc'] = 0.0
        else:
            # Multi-class AUROC (one-vs-rest)
            try:
                metrics['auroc'] = roc_auc_score(
                    targets, predictions, multi_class='ovr', average='macro'
                )
            except:
                metrics['auroc'] = 0.0
        
        # AUPRC
        try:
            if self.num_classes == 2:
                precision, recall, _ = precision_recall_curve(targets, predictions[:, 1])
                metrics['auprc'] = auc(recall, precision)
            else:
                # Multi-class AUPRC (macro average)
                auprc_scores = []
                for i in range(self.num_classes):
                    binary_targets = (targets == i).astype(int)
                    if len(np.unique(binary_targets)) > 1:  # Check if class exists
                        precision, recall, _ = precision_recall_curve(
                            binary_targets, predictions[:, i]
                        )
                        auprc_scores.append(auc(recall, precision))
                metrics['auprc'] = np.mean(auprc_scores) if auprc_scores else 0.0
        except:
            metrics['auprc'] = 0.0
        
        return metrics


class FewShotEvaluator:
    """Evaluator for few-shot learning tasks."""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        device: torch.device,
        n_way: int = 5,
        k_shot: int = 1
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
            n_way: Number of classes per task.
            k_shot: Number of support examples per class.
        """
        self.model = model
        self.device = device
        self.n_way = n_way
        self.k_shot = k_shot
        self.metrics = MetaLearningMetrics(n_way)
    
    def evaluate_task(
        self, 
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        query_labels: torch.Tensor,
        adaptation_steps: int = 5
    ) -> Dict[str, float]:
        """Evaluate model on a single few-shot task.
        
        Args:
            support_data: Support set data.
            support_labels: Support set labels.
            query_data: Query set data.
            query_labels: Query set labels.
            adaptation_steps: Number of adaptation steps.
            
        Returns:
            Dictionary of metrics for this task.
        """
        self.model.eval()
        
        # Move data to device
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        query_data = query_data.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Adapt model to support set
        adapted_model = self._adapt_to_support(
            support_data, support_labels, adaptation_steps
        )
        
        # Evaluate on query set
        with torch.no_grad():
            query_logits = adapted_model(query_data)
            query_loss = F.cross_entropy(query_logits, query_labels)
            
            # Compute predictions
            _, predictions = torch.max(query_logits, 1)
            
            # Update metrics
            self.metrics.update(predictions, query_labels, query_loss.item())
        
        return self.metrics.compute()
    
    def _adapt_to_support(
        self, 
        support_data: torch.Tensor, 
        support_labels: torch.Tensor,
        adaptation_steps: int
    ) -> torch.nn.Module:
        """Adapt model to support set.
        
        Args:
            support_data: Support set data.
            support_labels: Support set labels.
            adaptation_steps: Number of adaptation steps.
            
        Returns:
            Adapted model.
        """
        # Clone model for adaptation
        adapted_model = type(self.model)(**self._get_model_kwargs())
        adapted_model.load_state_dict(self.model.state_dict())
        adapted_model.to(self.device)
        adapted_model.train()
        
        # Create optimizer for adaptation
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
        
        # Adapt to support set
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            logits = adapted_model(support_data)
            loss = F.cross_entropy(logits, support_labels)
            loss.backward()
            optimizer.step()
        
        adapted_model.eval()
        return adapted_model
    
    def _get_model_kwargs(self) -> Dict:
        """Get model initialization kwargs.
        
        Returns:
            Dictionary of model kwargs.
        """
        # This is a simplified version - in practice, you'd need to
        # store the original model kwargs
        return {
            'input_channels': 1,
            'num_classes': self.n_way,
            'hidden_dim': 128
        }
    
    def evaluate_dataset(
        self, 
        dataloader: torch.utils.data.DataLoader,
        num_tasks: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on entire dataset.
        
        Args:
            dataloader: DataLoader with few-shot tasks.
            num_tasks: Number of tasks to evaluate (None for all).
            
        Returns:
            Dictionary of average metrics.
        """
        self.metrics.reset()
        
        task_count = 0
        for batch in dataloader:
            if num_tasks is not None and task_count >= num_tasks:
                break
            
            # Process each task in the batch
            batch_size = batch['support_data'].size(0)
            for i in range(batch_size):
                self.evaluate_task(
                    batch['support_data'][i],
                    batch['support_labels'][i],
                    batch['query_data'][i],
                    batch['query_labels'][i]
                )
                task_count += 1
                
                if num_tasks is not None and task_count >= num_tasks:
                    break
        
        return self.metrics.compute()


def compute_confidence_intervals(
    scores: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute confidence intervals for evaluation scores.
    
    Args:
        scores: List of scores.
        confidence: Confidence level.
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    
    # Use t-distribution for small samples
    if len(scores) < 30:
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        margin = t_val * std / np.sqrt(len(scores))
    else:
        # Use normal distribution for large samples
        z_val = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
        margin = z_val * std / np.sqrt(len(scores))
    
    return mean, mean - margin, mean + margin
