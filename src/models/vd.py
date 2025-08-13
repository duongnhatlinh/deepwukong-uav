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


