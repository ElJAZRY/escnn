"""
E(2)-Equivariant ResNet Implementation

This module provides E(2)-equivariant versions of ResNet architectures (ResNet-18, 34, 50, 101, 152).
The models are equivariant to rotations and reflections in the E(2) group, making them suitable
for tasks where the input data has rotational symmetry (e.g., satellite imagery, medical imaging,
microscopy).

The implementation follows the original ResNet architecture from:
    "Deep Residual Learning for Image Recognition" (He et al., 2016)
    https://arxiv.org/abs/1512.03385

With E(2)-equivariance based on:
    "General E(2)-Equivariant Steerable CNNs" (Weiler & Cesa, 2019)
    https://arxiv.org/abs/1911.08251

Key Features:
- BasicBlock for ResNet-18 and ResNet-34
- Bottleneck block for ResNet-50, ResNet-101, and ResNet-152
- Configurable rotation equivariance (C_N for N discrete rotations)
- Configurable reflection equivariance (D_N for dihedral group)
- Optional restriction to subgroups through the network
- Support for different input image sizes

Example Usage:
    >>> from e2resnet import e2_resnet18, e2_resnet50
    >>>
    >>> # ResNet-18 with D_8 equivariance (8 rotations + reflections)
    >>> model = e2_resnet18(num_classes=10, N=8, flip=True)
    >>>
    >>> # ResNet-50 with C_4 equivariance (4 rotations only)
    >>> model = e2_resnet50(num_classes=100, N=4, flip=False)
    >>>
    >>> # Forward pass
    >>> x = torch.randn(4, 3, 224, 224)
    >>> y = model(x)

Author: Based on escnn library by Gabriele Cesa
"""

from typing import Tuple, List, Optional, Union, Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import escnn.nn as enn
from escnn.nn import init
from escnn import gspaces


__all__ = [
    # Main ResNet class
    "E2ResNet",
    # BasicBlock models
    "e2_resnet18",
    "e2_resnet34",
    # Bottleneck models
    "e2_resnet50",
    "e2_resnet101",
    "e2_resnet152",
    # Pre-configured variants with specific symmetries
    "e2_resnet18_d8",
    "e2_resnet18_c8",
    "e2_resnet18_d4",
    "e2_resnet18_c4",
    "e2_resnet50_d8",
    "e2_resnet50_c8",
    "e2_resnet50_d4",
    "e2_resnet50_c4",
]


# =============================================================================
# Convolution Helper Functions
# =============================================================================

def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 2,
            padding: int = 3, bias: bool = False) -> enn.R2Conv:
    """7x7 E(2)-equivariant convolution with padding."""
    return enn.R2Conv(
        in_type, out_type, 7,
        stride=stride,
        padding=padding,
        bias=bias,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 1,
            padding: int = 2, bias: bool = False) -> enn.R2Conv:
    """5x5 E(2)-equivariant convolution with padding."""
    return enn.R2Conv(
        in_type, out_type, 5,
        stride=stride,
        padding=padding,
        bias=bias,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 1,
            padding: int = 1, bias: bool = False) -> enn.R2Conv:
    """3x3 E(2)-equivariant convolution with padding."""
    return enn.R2Conv(
        in_type, out_type, 3,
        stride=stride,
        padding=padding,
        bias=bias,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride: int = 1,
            bias: bool = False) -> enn.R2Conv:
    """1x1 E(2)-equivariant convolution (point-wise)."""
    return enn.R2Conv(
        in_type, out_type, 1,
        stride=stride,
        padding=0,
        bias=bias,
        sigma=None,
        frequencies_cutoff=lambda r: 3 * r,
    )


# =============================================================================
# Field Type Builders
# =============================================================================

def regular_field_type(gspace: gspaces.GSpace, channels: int,
                       fixparams: bool = True) -> enn.FieldType:
    """
    Build a regular representation field type with the specified number of channels.

    Args:
        gspace: The geometric space defining symmetries
        channels: Desired number of output channels (before group dimension adjustment)
        fixparams: If True, scale channels to maintain parameter count similar to CNN

    Returns:
        FieldType with regular representations
    """
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        # Scale to maintain similar parameter count as standard CNN
        channels = int(channels * math.sqrt(N) / N)
    else:
        channels = channels // N

    channels = max(1, channels)
    return enn.FieldType(gspace, [gspace.regular_repr] * channels)


def trivial_field_type(gspace: gspaces.GSpace, channels: int,
                       fixparams: bool = True) -> enn.FieldType:
    """
    Build a trivial (scalar) field type with the specified number of channels.

    Args:
        gspace: The geometric space defining symmetries
        channels: Desired number of output channels
        fixparams: If True, scale channels to maintain parameter count similar to CNN

    Returns:
        FieldType with trivial representations
    """
    if fixparams:
        channels = int(channels * math.sqrt(gspace.fibergroup.order()))

    channels = max(1, channels)
    return enn.FieldType(gspace, [gspace.trivial_repr] * channels)


# =============================================================================
# ResNet Building Blocks
# =============================================================================

class BasicBlock(enn.EquivariantModule):
    """
    Basic residual block for E(2)-equivariant ResNet (used in ResNet-18 and ResNet-34).

    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU -> out
        |                                              ^
        +------------------- shortcut -----------------+

    Args:
        in_type: Input field type
        out_type: Output field type
        stride: Stride for the first convolution (for downsampling)
        downsample: Optional downsampling module for the shortcut
    """

    expansion: int = 1

    def __init__(
        self,
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        stride: int = 1,
        downsample: Optional[enn.EquivariantModule] = None,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # Determine convolution size based on rotation order
        # Use larger kernels for finer rotation discretization to capture more frequencies
        rotations = in_type.gspace.fibergroup.order()
        if hasattr(in_type.gspace, 'rotations_order'):
            rotations = in_type.gspace.rotations_order

        if rotations in [0, 1, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5

        # First convolution
        self.conv1 = conv(in_type, out_type, stride=stride)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu1 = enn.ReLU(out_type, inplace=True)

        # Second convolution
        self.conv2 = conv(out_type, out_type)
        self.bn2 = enn.InnerBatchNorm(out_type)

        # Shortcut connection
        self.downsample = downsample
        self.relu2 = enn.ReLU(out_type, inplace=True)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)

        return out

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        return input_shape


class Bottleneck(enn.EquivariantModule):
    """
    Bottleneck residual block for E(2)-equivariant ResNet (used in ResNet-50, 101, 152).

    Architecture:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+) -> ReLU -> out
        |                                                                        ^
        +---------------------------- shortcut ----------------------------------+

    The bottleneck design reduces computational cost by:
    1. Reducing dimensions with 1x1 conv
    2. Processing in lower dimension with 3x3 conv
    3. Expanding back with 1x1 conv

    Args:
        in_type: Input field type
        mid_type: Middle (bottleneck) field type
        out_type: Output field type (should be expansion * mid_type channels)
        stride: Stride for the 3x3 convolution (for downsampling)
        downsample: Optional downsampling module for the shortcut
    """

    expansion: int = 4

    def __init__(
        self,
        in_type: enn.FieldType,
        mid_type: enn.FieldType,
        out_type: enn.FieldType,
        stride: int = 1,
        downsample: Optional[enn.EquivariantModule] = None,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # Determine 3x3 convolution size based on rotation order
        rotations = in_type.gspace.fibergroup.order()
        if hasattr(in_type.gspace, 'rotations_order'):
            rotations = in_type.gspace.rotations_order

        if rotations in [0, 1, 2, 4]:
            conv_mid = conv3x3
        else:
            conv_mid = conv5x5

        # 1x1 reduce
        self.conv1 = conv1x1(in_type, mid_type)
        self.bn1 = enn.InnerBatchNorm(mid_type)
        self.relu1 = enn.ReLU(mid_type, inplace=True)

        # 3x3 (or 5x5) convolution
        self.conv2 = conv_mid(mid_type, mid_type, stride=stride)
        self.bn2 = enn.InnerBatchNorm(mid_type)
        self.relu2 = enn.ReLU(mid_type, inplace=True)

        # 1x1 expand
        self.conv3 = conv1x1(mid_type, out_type)
        self.bn3 = enn.InnerBatchNorm(out_type)

        # Shortcut connection
        self.downsample = downsample
        self.relu3 = enn.ReLU(out_type, inplace=True)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu3(out)

        return out

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        return input_shape


# =============================================================================
# E(2)-Equivariant ResNet
# =============================================================================

class E2ResNet(nn.Module):
    """
    E(2)-Equivariant ResNet.

    This model implements ResNet with E(2) group equivariance, meaning it is
    equivariant to 2D rotations and (optionally) reflections.

    Args:
        block: Block type to use (BasicBlock or Bottleneck)
        layers: Number of blocks in each of the 4 stages
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance (e.g., 4, 8, 16)
        flip: If True, model is also equivariant to reflections (D_N group)
              If False, model is only rotation equivariant (C_N group)
        restrict: Restriction level - controls when to reduce symmetry:
                  0: No restriction (full equivariance throughout)
                  1: Restrict in layer3 (reduce symmetry in later layers)
                  2: Restrict in layer2 and layer3
                  3: Restrict progressively in layer2, layer3, and layer4
        in_channels: Number of input channels (default: 3 for RGB)
        base_width: Base width of the network (default: 64)
        fixparams: If True, scale channel widths to maintain similar
                   parameter count as standard CNN
        initialize: If True, apply proper weight initialization

    Example:
        >>> model = E2ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, N=8, flip=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> y = model(x)  # Shape: (4, 10)
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        N: int = 8,
        flip: bool = True,
        restrict: int = 0,
        in_channels: int = 3,
        base_width: int = 64,
        fixparams: bool = True,
        initialize: bool = True,
    ):
        super().__init__()

        self._N = N
        self._flip = flip
        self._restrict = restrict
        self._fixparams = fixparams
        self._base_width = base_width
        self._block = block

        # Validate restriction level
        assert restrict in [0, 1, 2, 3], "restrict must be 0, 1, 2, or 3"

        # Initialize the geometric space based on symmetry configuration
        if flip:
            if N > 1:
                self.gspace = gspaces.flipRot2dOnR2(N)
            else:
                self.gspace = gspaces.flip2dOnR2()
        else:
            if N > 1:
                self.gspace = gspaces.rot2dOnR2(N)
            else:
                self.gspace = gspaces.trivialOnR2()

        # Input type: trivial representation for RGB channels
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_channels)

        # Initial feature type after stem convolution
        self._current_type = regular_field_type(self.gspace, base_width, fixparams)

        # Stem: 7x7 conv, BN, ReLU, MaxPool
        self.conv1 = conv7x7(self.in_type, self._current_type, stride=2, padding=3)
        self.bn1 = enn.InnerBatchNorm(self._current_type)
        self.relu = enn.ReLU(self._current_type, inplace=True)
        self.maxpool = enn.PointwiseMaxPool(self._current_type, kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(block, base_width, layers[0])

        # Optional restriction before layer2
        if restrict >= 3:
            self.restrict1 = self._make_restriction(N // 2 if N > 1 else 1)
        else:
            self.restrict1 = lambda x: x

        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)

        # Optional restriction before layer3
        if restrict >= 2:
            new_N = N // 4 if restrict == 3 else N // 2
            new_N = max(1, new_N)
            self.restrict2 = self._make_restriction(new_N)
        else:
            self.restrict2 = lambda x: x

        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)

        # Optional restriction before layer4
        if restrict >= 1:
            self.restrict3 = self._make_restriction(1)  # Restrict to trivial
        else:
            self.restrict3 = lambda x: x

        # Final layer optionally outputs trivial features for classification
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2,
                                       to_trivial=(restrict > 0))

        # Final batch normalization
        self.bn_final = enn.InnerBatchNorm(self._current_type)
        self.relu_final = enn.ReLU(self._current_type, inplace=True)

        # Global average pooling (done on raw tensor)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer
        self.fc = nn.Linear(self._current_type.size, num_classes)

        # Weight initialization
        if initialize:
            self._initialize_weights()

    def _make_restriction(self, new_N: int) -> enn.SequentialModule:
        """Create a restriction layer to reduce equivariance to a subgroup."""
        layers = []

        if new_N == 1:
            subgroup_id = (0, 1) if self._flip else 1
        else:
            subgroup_id = (0, new_N) if self._flip else new_N

        layers.append(enn.RestrictionModule(self._current_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))

        self._current_type = layers[-1].out_type
        self.gspace = self._current_type.gspace

        return enn.SequentialModule(*layers)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        blocks: int,
        stride: int = 1,
        to_trivial: bool = False,
    ) -> enn.SequentialModule:
        """Build a ResNet stage with multiple blocks."""
        layers = []

        # Determine output type
        expansion = block.expansion

        if to_trivial:
            out_type = trivial_field_type(self.gspace, channels * expansion, self._fixparams)
        else:
            out_type = regular_field_type(self.gspace, channels * expansion, self._fixparams)

        # Downsample shortcut if needed
        downsample = None
        if stride != 1 or self._current_type != out_type:
            downsample = enn.SequentialModule(
                conv1x1(self._current_type, out_type, stride=stride),
                enn.InnerBatchNorm(out_type),
            )

        # First block (with potential downsampling)
        if block == BasicBlock:
            layers.append(
                block(
                    in_type=self._current_type,
                    out_type=out_type,
                    stride=stride,
                    downsample=downsample,
                )
            )
        else:  # Bottleneck
            mid_type = regular_field_type(self.gspace, channels, self._fixparams)
            layers.append(
                block(
                    in_type=self._current_type,
                    mid_type=mid_type,
                    out_type=out_type,
                    stride=stride,
                    downsample=downsample,
                )
            )

        self._current_type = out_type

        # Remaining blocks
        for _ in range(1, blocks):
            if block == BasicBlock:
                layers.append(
                    block(
                        in_type=self._current_type,
                        out_type=self._current_type,
                    )
                )
            else:  # Bottleneck
                mid_type = regular_field_type(self.gspace, channels, self._fixparams)
                layers.append(
                    block(
                        in_type=self._current_type,
                        mid_type=mid_type,
                        out_type=self._current_type,
                    )
                )

        return enn.SequentialModule(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                # Use generalized He initialization for equivariant convolutions
                init.generalized_he_init(module.weights, module.basisexpansion)
            elif isinstance(module, (nn.BatchNorm2d, enn.InnerBatchNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Wrap input in GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages with optional restrictions
        x = self.layer1(x)
        x = self.restrict1(x)

        x = self.layer2(x)
        x = self.restrict2(x)

        x = self.layer3(x)
        x = self.restrict3(x)

        x = self.layer4(x)

        # Final normalization
        x = self.bn_final(x)
        x = self.relu_final(x)

        # Extract tensor for pooling and FC
        x = x.tensor

        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> List[enn.GeometricTensor]:
        """
        Extract intermediate features from all stages.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            List of GeometricTensors from each stage
        """
        features = []

        x = enn.GeometricTensor(x, self.in_type)

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        features.append(x)
        x = self.restrict1(x)

        x = self.layer2(x)
        features.append(x)
        x = self.restrict2(x)

        x = self.layer3(x)
        features.append(x)
        x = self.restrict3(x)

        x = self.layer4(x)
        features.append(x)

        return features


# =============================================================================
# Factory Functions
# =============================================================================

def e2_resnet18(num_classes: int = 1000, N: int = 8, flip: bool = True,
                restrict: int = 0, **kwargs) -> E2ResNet:
    """
    E(2)-Equivariant ResNet-18.

    Args:
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance
        flip: If True, include reflection equivariance
        restrict: Restriction level (0-3)
        **kwargs: Additional arguments passed to E2ResNet
    """
    return E2ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                    N=N, flip=flip, restrict=restrict, **kwargs)


def e2_resnet34(num_classes: int = 1000, N: int = 8, flip: bool = True,
                restrict: int = 0, **kwargs) -> E2ResNet:
    """
    E(2)-Equivariant ResNet-34.

    Args:
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance
        flip: If True, include reflection equivariance
        restrict: Restriction level (0-3)
        **kwargs: Additional arguments passed to E2ResNet
    """
    return E2ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                    N=N, flip=flip, restrict=restrict, **kwargs)


def e2_resnet50(num_classes: int = 1000, N: int = 8, flip: bool = True,
                restrict: int = 0, **kwargs) -> E2ResNet:
    """
    E(2)-Equivariant ResNet-50.

    Args:
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance
        flip: If True, include reflection equivariance
        restrict: Restriction level (0-3)
        **kwargs: Additional arguments passed to E2ResNet
    """
    return E2ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                    N=N, flip=flip, restrict=restrict, **kwargs)


def e2_resnet101(num_classes: int = 1000, N: int = 8, flip: bool = True,
                 restrict: int = 0, **kwargs) -> E2ResNet:
    """
    E(2)-Equivariant ResNet-101.

    Args:
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance
        flip: If True, include reflection equivariance
        restrict: Restriction level (0-3)
        **kwargs: Additional arguments passed to E2ResNet
    """
    return E2ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes,
                    N=N, flip=flip, restrict=restrict, **kwargs)


def e2_resnet152(num_classes: int = 1000, N: int = 8, flip: bool = True,
                 restrict: int = 0, **kwargs) -> E2ResNet:
    """
    E(2)-Equivariant ResNet-152.

    Args:
        num_classes: Number of output classes
        N: Number of discrete rotations for equivariance
        flip: If True, include reflection equivariance
        restrict: Restriction level (0-3)
        **kwargs: Additional arguments passed to E2ResNet
    """
    return E2ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes,
                    N=N, flip=flip, restrict=restrict, **kwargs)


# =============================================================================
# Pre-configured Variants with Specific Symmetries
# =============================================================================

# ResNet-18 variants
def e2_resnet18_d8(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-18 with D_8 equivariance (8 rotations + reflections)."""
    return e2_resnet18(num_classes=num_classes, N=8, flip=True, **kwargs)


def e2_resnet18_c8(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-18 with C_8 equivariance (8 rotations only)."""
    return e2_resnet18(num_classes=num_classes, N=8, flip=False, **kwargs)


def e2_resnet18_d4(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-18 with D_4 equivariance (4 rotations + reflections, i.e., 90° symmetry)."""
    return e2_resnet18(num_classes=num_classes, N=4, flip=True, **kwargs)


def e2_resnet18_c4(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-18 with C_4 equivariance (4 rotations only, i.e., 90° symmetry)."""
    return e2_resnet18(num_classes=num_classes, N=4, flip=False, **kwargs)


# ResNet-50 variants
def e2_resnet50_d8(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-50 with D_8 equivariance (8 rotations + reflections)."""
    return e2_resnet50(num_classes=num_classes, N=8, flip=True, **kwargs)


def e2_resnet50_c8(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-50 with C_8 equivariance (8 rotations only)."""
    return e2_resnet50(num_classes=num_classes, N=8, flip=False, **kwargs)


def e2_resnet50_d4(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-50 with D_4 equivariance (4 rotations + reflections, i.e., 90° symmetry)."""
    return e2_resnet50(num_classes=num_classes, N=4, flip=True, **kwargs)


def e2_resnet50_c4(num_classes: int = 1000, **kwargs) -> E2ResNet:
    """ResNet-50 with C_4 equivariance (4 rotations only, i.e., 90° symmetry)."""
    return e2_resnet50(num_classes=num_classes, N=4, flip=False, **kwargs)


# =============================================================================
# Testing and Demonstration
# =============================================================================

def test_equivariance(model: E2ResNet, input_size: int = 33,
                      rotations: bool = True, reflections: bool = True) -> None:
    """
    Test the equivariance properties of a model.

    Args:
        model: The E2ResNet model to test
        input_size: Size of the test images
        rotations: Whether to test rotation equivariance
        reflections: Whether to test reflection equivariance
    """
    model.eval()

    # Create test input
    x = torch.randn(2, 3, input_size, input_size)

    # Forward pass on original input
    with torch.no_grad():
        y = model(x)

    print("\nTesting Equivariance Properties:")
    print("=" * 50)

    if reflections:
        # Test vertical flip
        x_flip_v = x.flip(dims=[3])
        with torch.no_grad():
            y_flip_v = model(x_flip_v)
        is_invariant = torch.allclose(y, y_flip_v, atol=1e-5)
        print(f"Vertical reflection invariance:   {'✓ YES' if is_invariant else '✗ NO'}")

        # Test horizontal flip
        x_flip_h = x.flip(dims=[2])
        with torch.no_grad():
            y_flip_h = model(x_flip_h)
        is_invariant = torch.allclose(y, y_flip_h, atol=1e-5)
        print(f"Horizontal reflection invariance: {'✓ YES' if is_invariant else '✗ NO'}")

    if rotations:
        # Test 90° rotation
        x_rot90 = x.rot90(1, (2, 3))
        with torch.no_grad():
            y_rot90 = model(x_rot90)
        is_invariant = torch.allclose(y, y_rot90, atol=1e-5)
        print(f"90° rotation invariance:          {'✓ YES' if is_invariant else '✗ NO'}")

        # Test 180° rotation
        x_rot180 = x.rot90(2, (2, 3))
        with torch.no_grad():
            y_rot180 = model(x_rot180)
        is_invariant = torch.allclose(y, y_rot180, atol=1e-5)
        print(f"180° rotation invariance:         {'✓ YES' if is_invariant else '✗ NO'}")

    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E(2)-Equivariant ResNet")
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='Model architecture')
    parser.add_argument('--N', type=int, default=8,
                        help='Number of rotations for equivariance')
    parser.add_argument('--flip', action='store_true', default=True,
                        help='Enable reflection equivariance')
    parser.add_argument('--no-flip', dest='flip', action='store_false',
                        help='Disable reflection equivariance')
    parser.add_argument('--restrict', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Restriction level')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--test', action='store_true',
                        help='Run equivariance test')

    args = parser.parse_args()

    # Build model
    model_builders = {
        'resnet18': e2_resnet18,
        'resnet34': e2_resnet34,
        'resnet50': e2_resnet50,
        'resnet101': e2_resnet101,
        'resnet152': e2_resnet152,
    }

    print(f"\nBuilding E(2)-Equivariant {args.model.upper()}")
    print(f"  Rotations: {'C' if not args.flip else 'D'}_{args.N}")
    print(f"  Restriction level: {args.restrict}")
    print(f"  Num classes: {args.num_classes}")

    model = model_builders[args.model](
        num_classes=args.num_classes,
        N=args.N,
        flip=args.flip,
        restrict=args.restrict,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")

    # Run equivariance test
    if args.test:
        test_equivariance(model, input_size=33,
                         rotations=(args.N >= 4),
                         reflections=args.flip)

    print("\nDone!")
