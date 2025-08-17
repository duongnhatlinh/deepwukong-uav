from torch import nn
from omegaconf import DictConfig
import torch
from src.datas.samples import XFGBatch
import lightning as L
from typing import Dict, Any, List
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, EnhancedGraphConvEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop, AdamW
import torch.nn.functional as F
from src.metrics import Statistic
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary
from src.models.loss_functions import get_loss_function  # Import our custom loss functions


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

        # Initialize loss function
        self.loss_function = get_loss_function(config)
        
        # Initialize output lists for epoch aggregation
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_outputs = []
        
        # Gradient accumulation setup
        self.gradient_accumulation_steps = getattr(config.hyper_parameters, 'gradient_accumulation_steps', 1)
        self.accumulated_loss = 0.0
        self.accumulation_step = 0

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
        """
        Configure optimizer and learning rate scheduler
        Supports both legacy config format and new AdamW config
        """
        optimizer_name = self._config.hyper_parameters.optimizer
        learning_rate = self._config.hyper_parameters.learning_rate
        
        # Get weight_decay from config, default to 0 for backward compatibility
        weight_decay = getattr(self._config.hyper_parameters, 'weight_decay', 0)
        
        # Create optimizer with appropriate parameters
        if optimizer_name == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == "Adam":
            # For Adam, weight_decay is applied differently (L2 penalty on gradients)
            optimizer = Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            # For other optimizers, use the general approach
            optimizer_class = self._get_optimizer(optimizer_name)
            optimizer = optimizer_class(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Configure learning rate scheduler
        decay_gamma = getattr(self._config.hyper_parameters, 'decay_gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: decay_gamma ** epoch
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
        
        # Calculate loss and normalize by accumulation steps
        loss = self.loss_function(logits, batch.labels) / self.gradient_accumulation_steps
        
        # Accumulate loss for logging
        self.accumulated_loss += loss.item()
        self.accumulation_step += 1

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
        
        # Only log when we complete a full accumulation cycle or at the end of epoch
        if self.accumulation_step % self.gradient_accumulation_steps == 0:
            # Log the accumulated loss (already averaged)
            self.log("train_loss", self.accumulated_loss, on_step=True, on_epoch=False, 
                    batch_size=batch_size * self.gradient_accumulation_steps)
            self.log_dict(batch_metric, on_step=True, on_epoch=False, 
                         batch_size=batch_size * self.gradient_accumulation_steps)
            self.log("F1", batch_metric["train_f1"], prog_bar=True, logger=False, 
                    batch_size=batch_size * self.gradient_accumulation_steps)
            
            # Reset accumulated values
            self.accumulated_loss = 0.0
        
        # Store output for epoch end (use original loss for statistics)
        output = {"loss": loss * self.gradient_accumulation_steps, "statistic": statistic}
        self.training_step_outputs.append(output)
            
        return loss  # Return the scaled loss for backpropagation

    def validation_step(self, batch: XFGBatch, batch_idx: int) -> Dict[str, Any]:
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = self.loss_function(logits, batch.labels)

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
        loss = self.loss_function(logits, batch.labels)

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

    def on_train_epoch_start(self) -> None:
        """Reset accumulation at the start of each epoch"""
        self.accumulated_loss = 0.0
        self.accumulation_step = 0

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