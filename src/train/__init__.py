"""Training utilities for meta-learning."""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import learn2learn as l2l
import numpy as np
from pathlib import Path

from ..models import create_model, MAMLModel
from ..metrics import FewShotEvaluator, MetaLearningMetrics
from ..utils import EarlyStopping, create_checkpoint_dir


class MAMLTrainer:
    """Trainer for MAML meta-learning."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        log_dir: Optional[str] = None
    ):
        """Initialize MAML trainer.
        
        Args:
            model: Model to train.
            device: Device to train on.
            config: Training configuration.
            log_dir: Directory for logging.
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Create MAML wrapper
        self.maml = l2l.algorithms.MAML(
            model,
            lr=config['maml']['inner_lr'],
            first_order=config['maml'].get('first_order', False)
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10),
            min_delta=config['training'].get('min_delta', 0.001)
        )
        
        # Training state
        self.epoch = 0
        self.best_val_score = 0.0
        self.checkpoint_dir = create_checkpoint_dir(config['logging']['checkpoint_dir'])
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config.
        
        Returns:
            Configured optimizer.
        """
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['lr']
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.maml.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return optim.SGD(self.maml.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.maml.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config.
        
        Returns:
            Configured scheduler or None.
        """
        scheduler_name = self.config['training'].get('scheduler', '').lower()
        
        if not scheduler_name:
            return None
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        elif scheduler_name == 'step':
            step_size = self.config['training'].get('step_size', 30)
            gamma = self.config['training'].get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                patience=5, 
                factor=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        metrics = MetaLearningMetrics()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            support_data = batch['support_data'].to(self.device)
            support_labels = batch['support_labels'].to(self.device)
            query_data = batch['query_data'].to(self.device)
            query_labels = batch['query_labels'].to(self.device)
            
            # Meta-learning step
            learner = self.maml.clone()
            
            # Inner loop: adapt to support set
            for step in range(self.config['maml']['inner_steps']):
                support_logits = learner(support_data)
                support_loss = nn.functional.cross_entropy(support_logits, support_labels)
                learner.adapt(support_loss)
            
            # Outer loop: evaluate on query set
            query_logits = learner(query_data)
            query_loss = nn.functional.cross_entropy(query_logits, query_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            query_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += query_loss.item()
            num_batches += 1
            
            with torch.no_grad():
                _, predictions = torch.max(query_logits, 1)
                metrics.update(predictions, query_labels, query_loss.item())
            
            # Log progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                print(f'Epoch {self.epoch}, Batch {batch_idx}, Loss: {query_loss.item():.4f}')
        
        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        evaluator = FewShotEvaluator(self.model, self.device)
        
        val_metrics = evaluator.evaluate_dataset(
            val_loader, 
            num_tasks=self.config['evaluation']['num_eval_tasks']
        )
        
        return val_metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of training history.
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Starting training for {self.config['training']['epochs']} epochs...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Logging
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping(val_metrics['accuracy'], self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            # Save checkpoint
            if val_metrics['accuracy'] > self.best_val_score:
                self.best_val_score = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
            
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(is_best=False)
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')


class BaselineTrainer:
    """Trainer for baseline models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        log_dir: Optional[str] = None
    ):
        """Initialize baseline trainer.
        
        Args:
            model: Model to train.
            device: Device to train on.
            config: Training configuration.
            log_dir: Directory for logging.
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_score = 0.0
        self.checkpoint_dir = create_checkpoint_dir(config['logging']['checkpoint_dir'])
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['lr']
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_name = self.config['training'].get('scheduler', '').lower()
        
        if not scheduler_name:
            return None
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = MetaLearningMetrics()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            support_data = batch['support_data'].to(self.device)
            support_labels = batch['support_labels'].to(self.device)
            query_data = batch['query_data'].to(self.device)
            query_labels = batch['query_labels'].to(self.device)
            
            # Combine support and query for standard training
            all_data = torch.cat([support_data, query_data], dim=0)
            all_labels = torch.cat([support_labels, query_labels], dim=0)
            
            # Forward pass
            logits = self.model(all_data)
            loss = nn.functional.cross_entropy(logits, all_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            with torch.no_grad():
                _, predictions = torch.max(logits, 1)
                metrics.update(predictions, all_labels, loss.item())
        
        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        evaluator = FewShotEvaluator(self.model, self.device)
        
        val_metrics = evaluator.evaluate_dataset(
            val_loader, 
            num_tasks=self.config['evaluation']['num_eval_tasks']
        )
        
        return val_metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Train the model."""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Starting baseline training for {self.config['training']['epochs']} epochs...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Logging
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            if val_metrics['accuracy'] > self.best_val_score:
                self.best_val_score = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
