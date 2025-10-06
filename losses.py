import torch
import torch.nn as nn
import numpy as np

class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection
    Paper: "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks"

    Better than L1/L2 for small errors, provides stronger gradients for larger errors
    """
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

        # Constant C to make the function continuous
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) predicted landmarks
            target: (B, N, 2) ground truth landmarks
        Returns:
            loss: scalar
        """
        delta = (target - pred).abs()

        # Apply wing loss formula
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )

        return losses.mean()


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss - State-of-the-art for landmark detection
    Paper: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"

    Automatically adapts the loss shape based on the error magnitude
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) predicted landmarks
            target: (B, N, 2) ground truth landmarks
        Returns:
            loss: scalar
        """
        delta = (target - pred).abs()

        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * \
            (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )

        return losses.mean()


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss)
    Less sensitive to outliers than L2, smoother than L1
    """
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) predicted landmarks
            target: (B, N, 2) ground truth landmarks
        Returns:
            loss: scalar
        """
        diff = (pred - target).abs()

        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss: Wing Loss + MSE Loss
    Combines benefits of both losses
    """
    def __init__(self, wing_weight=0.7, mse_weight=0.3, omega=10, epsilon=2):
        super(CombinedLoss, self).__init__()
        self.wing_loss = WingLoss(omega=omega, epsilon=epsilon)
        self.mse_loss = nn.MSELoss()
        self.wing_weight = wing_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 2) predicted landmarks
            target: (B, N, 2) ground truth landmarks
        Returns:
            loss: scalar
        """
        wing = self.wing_loss(pred, target)
        mse = self.mse_loss(pred, target)

        return self.wing_weight * wing + self.mse_weight * mse


def get_loss_function(loss_type='wing', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: 'wing', 'adaptive_wing', 'smooth_l1', 'mse', 'combined'
        **kwargs: parameters for specific loss functions

    Returns:
        loss_fn: loss function
    """
    if loss_type == 'wing':
        return WingLoss(
            omega=kwargs.get('omega', 10),
            epsilon=kwargs.get('epsilon', 2)
        )
    elif loss_type == 'adaptive_wing':
        return AdaptiveWingLoss(
            omega=kwargs.get('omega', 14),
            theta=kwargs.get('theta', 0.5),
            epsilon=kwargs.get('epsilon', 1),
            alpha=kwargs.get('alpha', 2.1)
        )
    elif loss_type == 'smooth_l1':
        return SmoothL1Loss(beta=kwargs.get('beta', 1.0))
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'combined':
        return CombinedLoss(
            wing_weight=kwargs.get('wing_weight', 0.7),
            mse_weight=kwargs.get('mse_weight', 0.3),
            omega=kwargs.get('omega', 10),
            epsilon=kwargs.get('epsilon', 2)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
