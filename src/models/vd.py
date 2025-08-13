from torch import nn
from omegaconf import DictConfig
import torch
from src.datas.samples import XFGBatch
import lightning as L
from typing import Dict, Any, List
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop, AdamW
import torch.nn.functional as F
from src.metrics import Statistic
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary


class DeepWuKong(L.LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax,
        "AdamW": AdamW,
    }

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        hidden_size = config.classifier.hidden_size
        self._graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self._hidden_layers = nn.Sequential(*layers)
        self._classifier = nn.Linear(hidden_size, config.classifier.n_classes)
        
        # Initialize output lists for epoch aggregation
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []


    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch (Batch): [n_XFG (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_XFG, hidden size]
        graph_hid = self._graph_encoder(batch)
        hiddens = self._hidden_layers(graph_hid)
        # [n_XFG; n_classes]
        return self._classifier(hiddens)

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self._optimizers[self._config.hyper_parameters.optimizer](
            self.parameters(),
            lr=self._config.hyper_parameters.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self._config.hyper_parameters.decay_gamma ** epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


    def training_step(self, batch: XFGBatch, batch_idx: int) -> Dict[str, Any]:
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            
        # Log metrics with batch_size
        batch_size = batch.labels.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log_dict(batch_metric, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("F1", batch_metric["train_f1"], prog_bar=True, logger=False, batch_size=batch_size)
        
        # Store output for epoch end
        output = {"loss": loss, "statistic": statistic}
        self.training_step_outputs.append(output)
            
        return output

    def validation_step(self, batch: XFGBatch, batch_idx: int) -> Dict[str, Any]:
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            
        # Log loss immediately with batch_size
        batch_size = batch.labels.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Store output for epoch end
        output = {"loss": loss, "statistic": statistic}
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch: XFGBatch, batch_idx: int) -> Dict[str, Any]:
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            
        # Log loss immediately with batch_size
        batch_size = batch.labels.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Store output for epoch end
        output = {"loss": loss, "statistic": statistic}
        self.test_step_outputs.append(output)

        return output

    # ========== EPOCH END ==========
    def on_train_epoch_end(self) -> None:
        if not self.training_step_outputs:
            return
            
        # Calculate epoch statistics
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in self.training_step_outputs]
        )
        metrics = statistic.calculate_metrics(group="train")
        
        # Log epoch metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        if not self.validation_step_outputs:
            return
            
        # Calculate epoch statistics
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in self.validation_step_outputs]
        )
        metrics = statistic.calculate_metrics(group="val")
        
        # Log epoch metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
            
        # Calculate epoch statistics
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in self.test_step_outputs]
        )
        metrics = statistic.calculate_metrics(group="test")
        
        # Log epoch metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.test_step_outputs.clear()



    # def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
    #              pad_idx: int):
    #     super().__init__()
    #     self.save_hyperparameters()
    #     self._config = config
    #     hidden_size = config.classifier.hidden_size
    #     self._graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
    #                                                            pad_idx)

    #     # Enhanced classifier with residual connections
    #     classifier_layers = []
    #     input_size = config.gnn.hidden_size
        
    #     for i in range(config.classifier.n_hidden_layers):
    #         # Hidden layer
    #         classifier_layers.extend([
    #             nn.Linear(input_size, config.classifier.hidden_size),
    #             nn.LayerNorm(config.classifier.hidden_size) if config.classifier.get('use_layer_norm') else nn.BatchNorm1d(config.classifier.hidden_size),
    #             nn.GELU() if config.classifier.get('activation') == 'gelu' else nn.ReLU(),
    #             nn.Dropout(config.classifier.drop_out)
    #         ])
    #         input_size = config.classifier.hidden_size
        
    #     self.hidden_layers = nn.Sequential(*classifier_layers)
        
    #     # Output layer
    #     self.classifier = nn.Linear(config.classifier.hidden_size, config.classifier.n_classes)
        
    #     # Loss function setup
    #     self.setup_loss_functions()


    # def setup_loss_functions(self):
    #     """Setup advanced loss functions"""
    #     # Class weights for imbalanced data
    #     if self._config.hyper_parameters.get('use_class_weights'):
    #         # These should be calculated from actual data
    #         class_weights = torch.tensor([0.3, 2.7])  # Adjust based on your data
    #         self.register_buffer('class_weights', class_weights)
        
    #     # Focal loss parameters
    #     if self._config.hyper_parameters.get('use_focal_loss'):
    #         self.focal_alpha = self._config.hyper_parameters.focal_alpha
    #         self.focal_gamma = self._config.hyper_parameters.focal_gamma
        
    #     # Label smoothing
    #     if self._config.hyper_parameters.get('use_label_smoothing'):
    #         self.label_smoothing = self._config.hyper_parameters.label_smoothing

    # def compute_loss(self, logits, labels):
    #     """Compute enhanced loss with multiple techniques"""
        
    #     if self._config.hyper_parameters.get('use_focal_loss'):
    #         loss = self.focal_loss(logits, labels)
    #     else:
    #         # Standard cross entropy with optional class weights
    #         weight = getattr(self, 'class_weights', None)
    #         loss = F.cross_entropy(logits, labels, weight=weight)
        
    #     # Add label smoothing if enabled
    #     if self._config.hyper_parameters.get('use_label_smoothing'):
    #         loss = self.label_smoothing_loss(logits, labels, loss)
        
    #     return loss

    # def forward(self, batch: Batch) -> torch.Tensor:
    #     # Graph encoding
    #     graph_repr = self._graph_encoder(batch)
        
    #     # Classification
    #     hidden = self.hidden_layers(graph_repr)
    #     logits = self.classifier(hidden)
        
    #     return logits

    # def focal_loss(self, logits, labels):
    #     """Focal loss for handling class imbalance"""
    #     ce_loss = F.cross_entropy(logits, labels, reduction='none')
    #     pt = torch.exp(-ce_loss)
    #     focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
    #     return focal_loss.mean()
    
    # def label_smoothing_loss(self, logits, labels, base_loss):
    #     """Label smoothing regularization"""
    #     n_classes = logits.size(-1)
    #     log_probs = F.log_softmax(logits, dim=-1)
    #     nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
    #     smooth_loss = -log_probs.mean(dim=-1)
        
    #     loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
    #     return loss.mean()

    # def configure_optimizers(self):
    #     """Enhanced optimizer configuration"""
    #     # Different learning rates for different components
    #     encoder_params = list(self._graph_encoder.parameters())
    #     classifier_params = list(self.hidden_layers.parameters()) + list(self.classifier.parameters())
        
    #     if self._config.hyper_parameters.optimizer == "AdamW":
    #         optimizer = torch.optim.AdamW([
    #             {'params': encoder_params, 'lr': self._config.hyper_parameters.learning_rate * 0.8},
    #             {'params': classifier_params, 'lr': self._config.hyper_parameters.learning_rate}
    #         ], weight_decay=self._config.hyper_parameters.weight_decay)
    #     else:
    #         optimizer = torch.optim.Adam(self.parameters(), 
    #                                    lr=self._config.hyper_parameters.learning_rate,
    #                                    weight_decay=self._config.hyper_parameters.weight_decay)
        
    #     # Learning rate scheduler
    #     if self._config.hyper_parameters.get('use_lr_scheduler'):
    #         if self._config.hyper_parameters.scheduler_type == "cosine_annealing_warm_restarts":
    #             scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #                 optimizer,
    #                 T_0=self._config.hyper_parameters.t_0,
    #                 T_mult=self._config.hyper_parameters.t_mult,
    #                 eta_min=self._config.hyper_parameters.eta_min
    #             )
    #         else:
    #             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                 optimizer, 
    #                 mode='max',
    #                 patience=self._config.hyper_parameters.get('patience_lr', 5),
    #                 factor=self._config.hyper_parameters.get('factor_lr', 0.5)
    #             )
            
    #         return {
    #             "optimizer": optimizer,
    #             "lr_scheduler": {
    #                 "scheduler": scheduler,
    #                 "monitor": "val_f1",
    #                 "frequency": 1
    #             }
    #         }
        
    #     return optimizer
    
    # def training_step(self, batch, batch_idx):
        
    #     logits = self(batch.graphs)
    #     loss = self.compute_loss(logits, batch.labels)
        
    #     # Get batch size
    #     batch_size = batch.labels.size(0)
        
    #     # Metrics calculation
    #     with torch.no_grad():
    #         _, preds = logits.max(dim=1)
    #         statistic = Statistic().calculate_statistic(batch.labels, preds, 2)
    #         metrics = statistic.calculate_metrics(group="train")
            
    #         # Per-class accuracy
    #         for class_id in [0, 1]:
    #             mask = (batch.labels == class_id)
    #             if mask.sum() > 0:
    #                 class_acc = (preds[mask] == class_id).float().mean()
    #                 self.log(f"train_class_{class_id}_acc", class_acc, batch_size=batch_size)
            
    #         # Log metrics with batch_size
    #         self.log_dict(metrics, on_step=True, on_epoch=False, batch_size=batch_size)
    #         self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        
    #     return {"loss": loss, "statistic": statistic}
    
    # def validation_step(self, batch: XFGBatch, batch_idx: int) -> torch.Tensor:
    #     """Validation step with enhanced metrics"""
    #     # [n_XFG; n_classes]
    #     logits = self(batch.graphs)
    #     loss = self.compute_loss(logits, batch.labels)

    #     with torch.no_grad():
    #         _, preds = logits.max(dim=1)
    #         statistic = Statistic().calculate_statistic(
    #             batch.labels,
    #             preds,
    #             2,
    #         )
    #         batch_metric = statistic.calculate_metrics(group="val")
            
    #         # Get batch size for logging
    #         batch_size = batch.labels.size(0)
            
    #         # Per-class accuracy for debugging
    #         for class_id in [0, 1]:
    #             mask = (batch.labels == class_id)
    #             if mask.sum() > 0:
    #                 class_acc = (preds[mask] == class_id).float().mean()
    #                 self.log(f"val_class_{class_id}_acc", class_acc, 
    #                         on_step=False, on_epoch=True, batch_size=batch_size)

    #     # Log metrics with batch_size
    #     self.log("val_loss", loss, on_step=False, on_epoch=True, 
    #             prog_bar=True, batch_size=batch_size)
    #     self.log_dict(batch_metric, on_step=False, on_epoch=True, batch_size=batch_size)
        
    #     return {"loss": loss, "statistic": statistic}

    # def test_step(self, batch: XFGBatch, batch_idx: int) -> torch.Tensor:
    #     """Test step with comprehensive metrics"""
    #     # [n_XFG; n_classes]
    #     logits = self(batch.graphs)
    #     loss = self.compute_loss(logits, batch.labels)

    #     with torch.no_grad():
    #         _, preds = logits.max(dim=1)
    #         statistic = Statistic().calculate_statistic(
    #             batch.labels,
    #             preds,
    #             2,
    #         )
    #         batch_metric = statistic.calculate_metrics(group="test")
            
    #         # Get batch size for logging
    #         batch_size = batch.labels.size(0)
            
    #         # Per-class accuracy for final evaluation
    #         for class_id in [0, 1]:
    #             mask = (batch.labels == class_id)
    #             if mask.sum() > 0:
    #                 class_acc = (preds[mask] == class_id).float().mean()
    #                 self.log(f"test_class_{class_id}_acc", class_acc, 
    #                         on_step=False, on_epoch=True, batch_size=batch_size)

    #     # Log test metrics with batch_size
    #     self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
    #     self.log_dict(batch_metric, on_step=False, on_epoch=True, batch_size=batch_size)

    #     return {"loss": loss, "statistic": statistic}
    
    # def on_train_epoch_end(self) -> None:
    #     """Handle training epoch end - log additional metrics"""
    #     # Get current learning rate for monitoring
    #     if hasattr(self.trainer.optimizers[0], 'param_groups'):
    #         current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #         self.log("learning_rate", current_lr, on_epoch=True, batch_size=1)

    # def on_validation_epoch_end(self) -> None:
    #     """Handle validation epoch end - can add custom logic here"""
    #     # Optional: Save best model or additional validation logic
    #     pass

    # def on_test_epoch_end(self) -> None:
    #     """Handle test epoch end - log final summary"""
    #     # Optional: Print final test results summary
    #     pass

    # def predict_step(self, batch: XFGBatch, batch_idx: int) -> Dict[str, torch.Tensor]:
    #     """Prediction step for inference"""
    #     logits = self(batch.graphs)
    #     probs = F.softmax(logits, dim=1)
    #     _, preds = logits.max(dim=1)
        
    #     return {
    #         "predictions": preds,
    #         "probabilities": probs,
    #         "logits": logits
    #     }

    # def predict_vulnerability(self, graph_batch: Batch) -> Dict[str, Any]:
    #     """Predict vulnerability for a single graph or batch"""
    #     self.eval()
    #     with torch.no_grad():
    #         logits = self(graph_batch)
    #         probs = F.softmax(logits, dim=1)
    #         _, preds = logits.max(dim=1)
            
    #         # Get confidence scores
    #         confidence_scores = probs.max(dim=1)[0]
            
    #         return {
    #             "predictions": preds.cpu().numpy(),
    #             "probabilities": probs.cpu().numpy(),
    #             "confidence": confidence_scores.cpu().numpy(),
    #             "is_vulnerable": (preds == 1).cpu().numpy()
    #         }
        
    # def log_confusion_matrix(self, batch_labels: torch.Tensor, batch_preds: torch.Tensor, stage: str):
    #     """Log confusion matrix for better analysis"""
    #     from sklearn.metrics import confusion_matrix
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
        
    #     if self.trainer.logger and hasattr(self.trainer.logger, 'experiment'):
    #         # Convert to numpy
    #         labels_np = batch_labels.cpu().numpy()
    #         preds_np = batch_preds.cpu().numpy()
            
    #         # Calculate confusion matrix
    #         cm = confusion_matrix(labels_np, preds_np)
            
    #         # Create plot
    #         plt.figure(figsize=(8, 6))
    #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #         plt.title(f'Confusion Matrix - {stage}')
    #         plt.ylabel('True Label')
    #         plt.xlabel('Predicted Label')
            
    #         # Log to tensorboard
    #         if hasattr(self.trainer.logger.experiment, 'add_figure'):
    #             self.trainer.logger.experiment.add_figure(
    #                 f'{stage}_confusion_matrix', 
    #                 plt.gcf(), 
    #                 self.current_epoch
    #             )
    #         plt.close()

    # def log_class_distribution(self, batch_labels: torch.Tensor, stage: str):
    #     """Log class distribution for monitoring"""
    #     unique, counts = torch.unique(batch_labels, return_counts=True)
        
    #     for class_id, count in zip(unique.cpu().numpy(), counts.cpu().numpy()):
    #         self.log(f"{stage}_class_{class_id}_count", float(count), on_step=False, on_epoch=True)

    # def on_before_optimizer_step(self, optimizer):
    #     """Monitor gradients before optimizer step - FIXED signature"""
    #     # Log gradient norms for debugging
    #     if self.current_epoch % 10 == 0:  # Log every 10 epochs
    #         grad_norm = 0.0
    #         param_count = 0
            
    #         for name, param in self.named_parameters():
    #             if param.grad is not None:
    #                 param_norm = param.grad.data.norm(2)
    #                 grad_norm += param_norm.item() ** 2
    #                 param_count += 1
            
    #         if param_count > 0:
    #             grad_norm = grad_norm ** (1. / 2)
    #             # Add batch_size for gradient norm logging
    #             self.log("gradient_norm", grad_norm, on_step=True, on_epoch=False, 
    #                     batch_size=1)  # Gradient norm is scalar, so batch_size=1

# -----------------------------------------------
    