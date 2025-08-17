import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection" by Lin et al.
    Formula: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    
    Args:
        alpha (float or list): Weighting factor for rare class. 
                              If float, alpha for class 1, (1-alpha) for class 0
                              If list, [alpha_class_0, alpha_class_1]
        gamma (float): Focusing parameter. Higher gamma -> more focus on hard examples
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            # alpha for class 1 (vulnerable), (1-alpha) for class 0 (safe)
            self.alpha = torch.tensor([1-alpha, alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            raise TypeError('Alpha must be float or list')
            
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [N, C] where C=2 for binary classification
            targets: Ground truth labels [N] with values 0 or 1
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probabilities
        p = torch.exp(-ce_loss)
        
        # Move alpha to same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
            
        # Get alpha for each target
        alpha_t = self.alpha[targets]
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for class imbalance
    
    Args:
        pos_weight (float): Weight for positive class (vulnerable)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [N, C] where C=2
            targets: Ground truth labels [N] with values 0 or 1
        """
        # Calculate class weights
        weight = torch.ones(inputs.size(1), device=inputs.device)
        weight[1] = self.pos_weight  # Weight for vulnerable class
        
        return F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)


def get_loss_function(config):
    """
    Factory function to create loss function based on config
    
    Args:
        config: Configuration object with loss_function settings
        
    Returns:
        Loss function instance
    """
    if not hasattr(config, 'loss_function'):
        # Default to standard cross entropy if no loss_function config
        return nn.CrossEntropyLoss()
    
    loss_config = config.loss_function
    loss_type = loss_config.get('type', 'cross_entropy').lower()
    
    if loss_type == 'focal_loss':
        return FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'weighted_cross_entropy':
        return WeightedCrossEntropyLoss(
            pos_weight=loss_config.get('pos_weight', 4.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")