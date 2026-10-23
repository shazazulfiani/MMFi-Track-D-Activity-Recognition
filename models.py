"""Arsitektur model 2D-CNN dan CNN-DANN untuk MM-Fi Track D."""

import torch
import torch.nn as nn

NUM_CLASSES = 27
FEATURE_DIM = 128


class FeatureExtractor(nn.Module):
    """Mengubah fitur CSI menjadi representasi fitur berdimensi 128."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(36, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64, FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.projection(self.conv(x))


class CNN2D(nn.Module):
    """Model baseline untuk klasifikasi 27 aktivitas."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class GradientReversalFunction(torch.autograd.Function):
    """Membalik arah gradien untuk melatih fitur yang sulit membedakan domain."""

    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, coefficient: float = 1.0):
        return GradientReversalFunction.apply(x, coefficient)


class CNNDANN(nn.Module):
    """CNN-DANN dengan kepala klasifikasi aktivitas dan kepala domain."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x, grl_coefficient: float = 1.0):
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        reversed_features = self.grl(features, grl_coefficient)
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits
