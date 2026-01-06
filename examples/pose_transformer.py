"""
6DoF Pose Estimation Transformer with SO(3) and R(3) Group Mappings

This module implements a Vision Transformer (ViT) based architecture for 6DoF
pose estimation from 2D images. The model predicts both:
- 3D rotation in SO(3) (Special Orthogonal Group)
- 3D translation in R(3) (3D Euclidean space)

The architecture follows the Vision Transformer paradigm:
1. Split image into patches
2. Embed patches with positional encoding
3. Process through transformer encoder blocks
4. Map to SO(3) for rotation and R(3) for translation

Key Features:
- Multiple SO(3) representations: quaternions, 6D continuous, axis-angle, rotation matrix
- Geodesic loss on SO(3) for proper rotation learning
- Optional equivariant patch embedding using E(2) steerable convolutions
- Support for single object and multi-object pose estimation

References:
    - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    - "On the Continuity of Rotation Representations" (Zhou et al., CVPR 2019)
    - "General E(2)-Equivariant Steerable CNNs" (Weiler & Cesa, 2019)

Example Usage:
    >>> from pose_transformer import PoseTransformer, pose_loss
    >>>
    >>> # Create model
    >>> model = PoseTransformer(
    ...     image_size=224,
    ...     patch_size=16,
    ...     num_classes=1,  # Single object
    ...     rotation_repr='6d',  # 6D continuous representation
    ... )
    >>>
    >>> # Forward pass
    >>> images = torch.randn(4, 3, 224, 224)
    >>> rotation, translation = model(images)
    >>>
    >>> # Compute loss
    >>> loss = pose_loss(rotation, translation, gt_rotation, gt_translation,
    ...                  rotation_repr='6d')

Author: Based on escnn library patterns
"""

from typing import Tuple, List, Optional, Union, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional escnn imports for equivariant patch embedding
try:
    import escnn.nn as enn
    from escnn import gspaces
    ESCNN_AVAILABLE = True
except ImportError:
    ESCNN_AVAILABLE = False


__all__ = [
    # Main model
    "PoseTransformer",
    # Components
    "PatchEmbedding",
    "EquivariantPatchEmbedding",
    "TransformerEncoder",
    "SO3Head",
    "R3Head",
    # Loss functions
    "pose_loss",
    "geodesic_loss_quat",
    "geodesic_loss_rotmat",
    "rotation_6d_to_matrix",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "matrix_to_axis_angle",
]


# =============================================================================
# SO(3) Rotation Representations and Conversions
# =============================================================================

def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.

    The 6D representation is from "On the Continuity of Rotation Representations"
    (Zhou et al., CVPR 2019). It represents a rotation using two column vectors
    of the rotation matrix, and recovers the third via cross product.

    This representation is continuous and thus better for learning than
    quaternions or Euler angles.

    Args:
        rot_6d: Tensor of shape (..., 6) containing two 3D vectors

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Split into two 3D vectors
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into rotation matrix
    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.

    Quaternion format: [x, y, z, w] where w is the scalar part.

    Args:
        quaternion: Tensor of shape (..., 4)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Normalize quaternion
    quaternion = F.normalize(quaternion, dim=-1)

    x, y, z, w = quaternion.unbind(-1)

    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)

    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        matrix: Tensor of shape (..., 3, 3)

    Returns:
        Quaternions of shape (..., 4) in [x, y, z, w] format
    """
    batch_shape = matrix.shape[:-2]
    matrix = matrix.view(-1, 3, 3)

    # Compute quaternion components
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]

    quaternion = torch.zeros(matrix.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4 * w
    quaternion[mask, 3] = 0.25 * s
    quaternion[mask, 0] = (matrix[mask, 2, 1] - matrix[mask, 1, 2]) / s
    quaternion[mask, 1] = (matrix[mask, 0, 2] - matrix[mask, 2, 0]) / s
    quaternion[mask, 2] = (matrix[mask, 1, 0] - matrix[mask, 0, 1]) / s

    # Case 2: (m00 > m11) and (m00 > m22)
    mask = (~mask) & (matrix[:, 0, 0] > matrix[:, 1, 1]) & (matrix[:, 0, 0] > matrix[:, 2, 2])
    s = torch.sqrt(1.0 + matrix[mask, 0, 0] - matrix[mask, 1, 1] - matrix[mask, 2, 2]) * 2
    quaternion[mask, 3] = (matrix[mask, 2, 1] - matrix[mask, 1, 2]) / s
    quaternion[mask, 0] = 0.25 * s
    quaternion[mask, 1] = (matrix[mask, 0, 1] + matrix[mask, 1, 0]) / s
    quaternion[mask, 2] = (matrix[mask, 0, 2] + matrix[mask, 2, 0]) / s

    # Case 3: m11 > m22
    mask = (trace <= 0) & (~(matrix[:, 0, 0] > matrix[:, 1, 1]) | ~(matrix[:, 0, 0] > matrix[:, 2, 2])) & (matrix[:, 1, 1] > matrix[:, 2, 2])
    s = torch.sqrt(1.0 + matrix[mask, 1, 1] - matrix[mask, 0, 0] - matrix[mask, 2, 2]) * 2
    quaternion[mask, 3] = (matrix[mask, 0, 2] - matrix[mask, 2, 0]) / s
    quaternion[mask, 0] = (matrix[mask, 0, 1] + matrix[mask, 1, 0]) / s
    quaternion[mask, 1] = 0.25 * s
    quaternion[mask, 2] = (matrix[mask, 1, 2] + matrix[mask, 2, 1]) / s

    # Case 4: else
    mask = (trace <= 0) & (~(matrix[:, 0, 0] > matrix[:, 1, 1]) | ~(matrix[:, 0, 0] > matrix[:, 2, 2])) & ~(matrix[:, 1, 1] > matrix[:, 2, 2])
    s = torch.sqrt(1.0 + matrix[mask, 2, 2] - matrix[mask, 0, 0] - matrix[mask, 1, 1]) * 2
    quaternion[mask, 3] = (matrix[mask, 1, 0] - matrix[mask, 0, 1]) / s
    quaternion[mask, 0] = (matrix[mask, 0, 2] + matrix[mask, 2, 0]) / s
    quaternion[mask, 1] = (matrix[mask, 1, 2] + matrix[mask, 2, 1]) / s
    quaternion[mask, 2] = 0.25 * s

    return quaternion.view(*batch_shape, 4)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to axis-angle representation.

    Args:
        matrix: Tensor of shape (..., 3, 3)

    Returns:
        Axis-angle vectors of shape (..., 3) where ||v|| = angle
    """
    quaternion = matrix_to_quaternion(matrix)
    return quaternion_to_axis_angle(quaternion)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to axis-angle representation.

    Args:
        quaternion: Tensor of shape (..., 4) in [x, y, z, w] format

    Returns:
        Axis-angle vectors of shape (..., 3)
    """
    # Normalize
    quaternion = F.normalize(quaternion, dim=-1)

    # Ensure w >= 0 for unique representation
    quaternion = torch.where(
        quaternion[..., 3:4] < 0,
        -quaternion,
        quaternion
    )

    # Extract components
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]

    # Compute angle
    sin_half_angle = torch.norm(xyz, dim=-1, keepdim=True)
    cos_half_angle = w

    # Two cases: small angle vs normal
    # For small angles, use Taylor expansion
    small_angle = sin_half_angle.abs() < 1e-6

    # Normal case: angle = 2 * atan2(sin, cos)
    angle = 2.0 * torch.atan2(sin_half_angle, cos_half_angle)

    # Axis (normalized xyz)
    axis = xyz / (sin_half_angle + 1e-8)

    # Axis-angle = axis * angle
    axis_angle = axis * angle

    # For small angles, xyz ≈ axis * sin(angle/2) ≈ axis * angle/2
    axis_angle = torch.where(small_angle, 2.0 * xyz, axis_angle)

    return axis_angle


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to 3x3 rotation matrix using Rodrigues formula.

    Args:
        axis_angle: Tensor of shape (..., 3) where ||v|| = angle

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)

    # Rodrigues formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # Skew-symmetric matrix K
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)

    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    I = I.expand(*axis_angle.shape[:-1], 3, 3)

    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * torch.bmm(
        K.view(-1, 3, 3), K.view(-1, 3, 3)
    ).view(*axis_angle.shape[:-1], 3, 3)

    # Handle zero rotation
    zero_rot = angle.squeeze(-1) < 1e-8
    R = torch.where(zero_rot.unsqueeze(-1).unsqueeze(-1), I, R)

    return R


# =============================================================================
# Loss Functions for Pose Estimation
# =============================================================================

def geodesic_loss_quat(pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
    """
    Geodesic loss on SO(3) using quaternion representation.

    The geodesic distance on SO(3) is the angle of the rotation that takes
    one orientation to another: d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

    For quaternions: d(q1, q2) = 2 * arccos(|q1 · q2|)

    Args:
        pred_quat: Predicted quaternions (..., 4)
        gt_quat: Ground truth quaternions (..., 4)

    Returns:
        Geodesic loss (scalar)
    """
    # Normalize quaternions
    pred_quat = F.normalize(pred_quat, dim=-1)
    gt_quat = F.normalize(gt_quat, dim=-1)

    # Quaternion dot product (with absolute value due to double cover)
    dot = torch.abs((pred_quat * gt_quat).sum(dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)

    # Geodesic distance
    angle = 2.0 * torch.acos(dot)

    return angle.mean()


def geodesic_loss_rotmat(pred_mat: torch.Tensor, gt_mat: torch.Tensor) -> torch.Tensor:
    """
    Geodesic loss on SO(3) using rotation matrix representation.

    d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

    Args:
        pred_mat: Predicted rotation matrices (..., 3, 3)
        gt_mat: Ground truth rotation matrices (..., 3, 3)

    Returns:
        Geodesic loss (scalar)
    """
    # R1^T @ R2
    diff = torch.bmm(
        pred_mat.view(-1, 3, 3).transpose(-2, -1),
        gt_mat.view(-1, 3, 3)
    )

    # Trace
    trace = diff[:, 0, 0] + diff[:, 1, 1] + diff[:, 2, 2]

    # Clamp for numerical stability
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # Geodesic distance
    angle = torch.acos(cos_angle)

    return angle.mean()


def translation_loss(pred_trans: torch.Tensor, gt_trans: torch.Tensor,
                     loss_type: str = 'l2') -> torch.Tensor:
    """
    Translation loss in R(3).

    Args:
        pred_trans: Predicted translations (..., 3)
        gt_trans: Ground truth translations (..., 3)
        loss_type: 'l1', 'l2', or 'smooth_l1'

    Returns:
        Translation loss (scalar)
    """
    if loss_type == 'l1':
        return F.l1_loss(pred_trans, gt_trans)
    elif loss_type == 'l2':
        return F.mse_loss(pred_trans, gt_trans)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_trans, gt_trans)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def pose_loss(
    pred_rotation: torch.Tensor,
    pred_translation: torch.Tensor,
    gt_rotation: torch.Tensor,
    gt_translation: torch.Tensor,
    rotation_repr: str = '6d',
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
    translation_loss_type: str = 'l2',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined 6DoF pose loss.

    Args:
        pred_rotation: Predicted rotation in specified representation
        pred_translation: Predicted translation (B, 3)
        gt_rotation: Ground truth rotation (as rotation matrix (B, 3, 3))
        gt_translation: Ground truth translation (B, 3)
        rotation_repr: Representation of pred_rotation ('6d', 'quat', 'axis_angle', 'matrix')
        rotation_weight: Weight for rotation loss
        translation_weight: Weight for translation loss
        translation_loss_type: Type of translation loss

    Returns:
        Tuple of (total_loss, rotation_loss, translation_loss)
    """
    # Convert predicted rotation to matrix
    if rotation_repr == '6d':
        pred_mat = rotation_6d_to_matrix(pred_rotation)
    elif rotation_repr == 'quat':
        pred_mat = quaternion_to_matrix(pred_rotation)
    elif rotation_repr == 'axis_angle':
        pred_mat = axis_angle_to_matrix(pred_rotation)
    elif rotation_repr == 'matrix':
        pred_mat = pred_rotation
    else:
        raise ValueError(f"Unknown rotation representation: {rotation_repr}")

    # Compute losses
    rot_loss = geodesic_loss_rotmat(pred_mat, gt_rotation)
    trans_loss = translation_loss(pred_translation, gt_translation, translation_loss_type)

    # Combined loss
    total_loss = rotation_weight * rot_loss + translation_weight * trans_loss

    return total_loss, rot_loss, trans_loss


# =============================================================================
# Patch Embedding Modules
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Standard patch embedding for Vision Transformer.

    Splits image into non-overlapping patches and projects to embedding dimension.

    Args:
        image_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch projection using convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input size {H}x{W} doesn't match model {self.image_size}x{self.image_size}"

        # Project patches: (B, embed_dim, H/patch, W/patch)
        x = self.proj(x)

        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class EquivariantPatchEmbedding(nn.Module):
    """
    E(2)-Equivariant patch embedding using steerable convolutions.

    This embedding preserves rotational structure in the patch features,
    which can be beneficial for pose estimation tasks.

    Args:
        image_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        N: Number of rotations for E(2) equivariance
        flip: Include reflection equivariance
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        N: int = 8,
        flip: bool = False,
    ):
        super().__init__()

        if not ESCNN_AVAILABLE:
            raise ImportError("escnn is required for EquivariantPatchEmbedding")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Create gspace
        if flip:
            self.gspace = gspaces.flipRot2dOnR2(N) if N > 1 else gspaces.flip2dOnR2()
        else:
            self.gspace = gspaces.rot2dOnR2(N) if N > 1 else gspaces.trivialOnR2()

        # Input type: trivial for RGB
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_channels)

        # Intermediate type: regular representation
        group_order = self.gspace.fibergroup.order()
        hidden_channels = embed_dim // group_order
        self.hidden_type = enn.FieldType(
            self.gspace, [self.gspace.regular_repr] * hidden_channels
        )

        # Output type: trivial (for compatibility with transformer)
        self.out_type = enn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * embed_dim
        )

        # Equivariant convolutions
        self.conv1 = enn.R2Conv(
            self.in_type, self.hidden_type,
            kernel_size=patch_size // 2,
            stride=patch_size // 2,
            padding=0,
            bias=False,
        )
        self.bn1 = enn.InnerBatchNorm(self.hidden_type)
        self.relu1 = enn.ReLU(self.hidden_type, inplace=True)

        self.conv2 = enn.R2Conv(
            self.hidden_type, self.out_type,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        # Wrap in GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)

        # Equivariant feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        # Extract tensor and reshape
        x = x.tensor  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        return x


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# =============================================================================
# Pose Prediction Heads
# =============================================================================

class SO3Head(nn.Module):
    """
    Prediction head for SO(3) rotation.

    Supports multiple rotation representations:
    - '6d': 6D continuous representation (recommended)
    - 'quat': Quaternion [x, y, z, w]
    - 'axis_angle': Axis-angle (3D vector)
    - 'matrix': Direct 3x3 matrix (9D, then orthogonalized)

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        rotation_repr: Output rotation representation
        num_objects: Number of objects to predict poses for
    """

    REPR_DIMS = {
        '6d': 6,
        'quat': 4,
        'axis_angle': 3,
        'matrix': 9,
    }

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        rotation_repr: str = '6d',
        num_objects: int = 1,
    ):
        super().__init__()

        self.rotation_repr = rotation_repr
        self.num_objects = num_objects
        self.out_dim = self.REPR_DIMS[rotation_repr]

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_objects * self.out_dim),
        )

        # Initialize to identity rotation
        self._init_weights()

    def _init_weights(self):
        """Initialize to output identity rotation."""
        nn.init.zeros_(self.head[-1].weight)

        # Set bias for identity rotation
        with torch.no_grad():
            if self.rotation_repr == '6d':
                # First column [1,0,0], second column [0,1,0]
                identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
            elif self.rotation_repr == 'quat':
                # Identity quaternion [0, 0, 0, 1]
                identity = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            elif self.rotation_repr == 'axis_angle':
                # Zero rotation
                identity = torch.tensor([0, 0, 0], dtype=torch.float32)
            elif self.rotation_repr == 'matrix':
                # Flatten identity matrix
                identity = torch.eye(3).flatten()

            identity = identity.repeat(self.num_objects)
            self.head[-1].bias.data = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, in_features)

        Returns:
            Rotation in specified representation (B, num_objects, repr_dim)
            or (B, repr_dim) if num_objects == 1
        """
        out = self.head(x)

        if self.num_objects > 1:
            out = out.view(-1, self.num_objects, self.out_dim)

        # Post-process based on representation
        if self.rotation_repr == 'quat':
            out = F.normalize(out, dim=-1)

        return out

    def to_matrix(self, rotation: torch.Tensor) -> torch.Tensor:
        """Convert output to rotation matrix."""
        if self.rotation_repr == '6d':
            return rotation_6d_to_matrix(rotation)
        elif self.rotation_repr == 'quat':
            return quaternion_to_matrix(rotation)
        elif self.rotation_repr == 'axis_angle':
            return axis_angle_to_matrix(rotation)
        elif self.rotation_repr == 'matrix':
            # Orthogonalize via SVD
            mat = rotation.view(*rotation.shape[:-1], 3, 3)
            u, s, vh = torch.linalg.svd(mat)
            return u @ vh


class R3Head(nn.Module):
    """
    Prediction head for R(3) translation.

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_objects: Number of objects to predict translations for
        output_scale: Scale factor for translation output
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_objects: int = 1,
        output_scale: float = 1.0,
    ):
        super().__init__()

        self.num_objects = num_objects
        self.output_scale = output_scale

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_objects * 3),
        )

        # Initialize to zero translation
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, in_features)

        Returns:
            Translation (B, num_objects, 3) or (B, 3) if num_objects == 1
        """
        out = self.head(x) * self.output_scale

        if self.num_objects > 1:
            out = out.view(-1, self.num_objects, 3)

        return out


# =============================================================================
# Main Pose Transformer Model
# =============================================================================

class PoseTransformer(nn.Module):
    """
    Vision Transformer for 6DoF Pose Estimation.

    Takes 2D images as input and predicts 6DoF pose (3D rotation + 3D translation)
    by mapping features to SO(3) and R(3) groups.

    Architecture:
        Input Image -> Patch Embedding -> Positional Encoding ->
        Transformer Encoder -> [CLS] token -> SO(3) Head + R(3) Head -> 6DoF Pose

    Args:
        image_size: Input image size (square)
        patch_size: Size of image patches
        in_channels: Number of input channels
        embed_dim: Transformer embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        rotation_repr: Rotation representation ('6d', 'quat', 'axis_angle', 'matrix')
        num_objects: Number of objects to predict poses for
        translation_scale: Scale factor for translation output
        use_equivariant_embedding: Use E(2)-equivariant patch embedding
        equivariant_N: Number of rotations for equivariant embedding
        equivariant_flip: Include reflections in equivariant embedding

    Example:
        >>> model = PoseTransformer(
        ...     image_size=224,
        ...     patch_size=16,
        ...     rotation_repr='6d',
        ... )
        >>> images = torch.randn(4, 3, 224, 224)
        >>> rotation, translation = model(images)
        >>> print(rotation.shape)  # (4, 6)
        >>> print(translation.shape)  # (4, 3)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rotation_repr: str = '6d',
        num_objects: int = 1,
        translation_scale: float = 1.0,
        use_equivariant_embedding: bool = False,
        equivariant_N: int = 8,
        equivariant_flip: bool = False,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.rotation_repr = rotation_repr
        self.num_objects = num_objects

        # Patch embedding
        if use_equivariant_embedding and ESCNN_AVAILABLE:
            self.patch_embed = EquivariantPatchEmbedding(
                image_size, patch_size, in_channels, embed_dim,
                N=equivariant_N, flip=equivariant_flip,
            )
        else:
            self.patch_embed = PatchEmbedding(
                image_size, patch_size, in_channels, embed_dim,
            )

        num_patches = self.patch_embed.num_patches

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio, dropout,
        )

        # Pose prediction heads
        self.rotation_head = SO3Head(
            embed_dim, embed_dim // 3, rotation_repr, num_objects,
        )
        self.translation_head = R3Head(
            embed_dim, embed_dim // 3, num_objects, translation_scale,
        )

        # Initialize positional embeddings
        self._init_pos_embed()
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_pos_embed(self):
        """Initialize positional embeddings with sinusoidal pattern."""
        num_patches = self.patch_embed.num_patches

        # Use sinusoidal positional encoding
        pos = torch.arange(num_patches + 1).float()
        dim = torch.arange(self.embed_dim).float()

        # Standard sinusoidal encoding
        pos = pos.unsqueeze(1)
        dim = dim.unsqueeze(0)
        angles = pos / (10000 ** (2 * (dim // 2) / self.embed_dim))

        pe = torch.zeros(1, num_patches + 1, self.embed_dim)
        pe[0, :, 0::2] = torch.sin(angles[:, 0::2])
        pe[0, :, 1::2] = torch.cos(angles[:, 1::2])

        self.pos_embed.data = pe

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Tuple of:
                - rotation: Rotation in specified representation
                - translation: 3D translation vector
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.encoder(x)

        # Extract CLS token for prediction
        cls_output = x[:, 0]  # (B, embed_dim)

        # Predict pose
        rotation = self.rotation_head(cls_output)
        translation = self.translation_head(cls_output)

        return rotation, translation

    def get_rotation_matrix(self, rotation: torch.Tensor) -> torch.Tensor:
        """Convert rotation output to 3x3 matrix."""
        return self.rotation_head.to_matrix(rotation)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before pose heads."""
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)

        return x


# =============================================================================
# Pre-configured Model Variants
# =============================================================================

def pose_transformer_tiny(**kwargs) -> PoseTransformer:
    """Tiny Pose Transformer (5.7M params)."""
    return PoseTransformer(
        embed_dim=192, depth=12, num_heads=3, **kwargs
    )


def pose_transformer_small(**kwargs) -> PoseTransformer:
    """Small Pose Transformer (22M params)."""
    return PoseTransformer(
        embed_dim=384, depth=12, num_heads=6, **kwargs
    )


def pose_transformer_base(**kwargs) -> PoseTransformer:
    """Base Pose Transformer (86M params)."""
    return PoseTransformer(
        embed_dim=768, depth=12, num_heads=12, **kwargs
    )


def pose_transformer_large(**kwargs) -> PoseTransformer:
    """Large Pose Transformer (307M params)."""
    return PoseTransformer(
        embed_dim=1024, depth=24, num_heads=16, **kwargs
    )


# =============================================================================
# Testing and Demonstration
# =============================================================================

def test_rotation_conversions():
    """Test rotation representation conversions."""
    print("\nTesting Rotation Conversions:")
    print("=" * 50)

    # Random rotation matrix (via QR decomposition)
    A = torch.randn(4, 3, 3)
    Q, R = torch.linalg.qr(A)
    det = torch.det(Q)
    Q = Q * det.unsqueeze(-1).unsqueeze(-1).sign()  # Ensure det = 1

    # Test round-trips
    # Matrix -> Quaternion -> Matrix
    quat = matrix_to_quaternion(Q)
    Q_from_quat = quaternion_to_matrix(quat)
    error_quat = torch.norm(Q - Q_from_quat).item()
    print(f"Matrix -> Quaternion -> Matrix error: {error_quat:.2e}")

    # Matrix -> 6D -> Matrix
    rot_6d = Q[..., :2].flatten(-2)  # First two columns
    Q_from_6d = rotation_6d_to_matrix(rot_6d)
    error_6d = torch.norm(Q - Q_from_6d).item()
    print(f"Matrix -> 6D -> Matrix error: {error_6d:.2e}")

    # Matrix -> Axis-Angle -> Matrix
    axis_angle = matrix_to_axis_angle(Q)
    Q_from_aa = axis_angle_to_matrix(axis_angle)
    error_aa = torch.norm(Q - Q_from_aa).item()
    print(f"Matrix -> Axis-Angle -> Matrix error: {error_aa:.2e}")

    print("=" * 50)


def test_pose_transformer():
    """Test PoseTransformer model."""
    print("\nTesting Pose Transformer:")
    print("=" * 50)

    # Create model
    model = PoseTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        rotation_repr='6d',
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        rotation, translation = model(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Rotation shape: {tuple(rotation.shape)}")
    print(f"Translation shape: {tuple(translation.shape)}")

    # Convert rotation to matrix
    rot_matrix = model.get_rotation_matrix(rotation)
    print(f"Rotation matrix shape: {tuple(rot_matrix.shape)}")

    # Verify rotation matrix properties
    det = torch.det(rot_matrix)
    RtR = torch.bmm(rot_matrix.transpose(-2, -1), rot_matrix)
    identity_error = torch.norm(RtR - torch.eye(3)).item()

    print(f"Rotation determinants: {det.tolist()}")
    print(f"R^T R - I error: {identity_error:.2e}")

    # Test loss computation
    gt_rotation = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    gt_translation = torch.zeros(2, 3)

    total_loss, rot_loss, trans_loss = pose_loss(
        rotation, translation, gt_rotation, gt_translation,
        rotation_repr='6d'
    )
    print(f"\nLoss values:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Rotation: {rot_loss.item():.4f}")
    print(f"  Translation: {trans_loss.item():.4f}")

    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="6DoF Pose Estimation Transformer")
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model size variant')
    parser.add_argument('--rotation_repr', type=str, default='6d',
                        choices=['6d', 'quat', 'axis_angle', 'matrix'],
                        help='Rotation representation')
    parser.add_argument('--equivariant', action='store_true',
                        help='Use equivariant patch embedding')

    args = parser.parse_args()

    if args.test:
        test_rotation_conversions()
        test_pose_transformer()
    else:
        # Build and display model
        model_builders = {
            'tiny': pose_transformer_tiny,
            'small': pose_transformer_small,
            'base': pose_transformer_base,
            'large': pose_transformer_large,
        }

        print(f"\nBuilding Pose Transformer ({args.model})")
        print(f"  Rotation repr: {args.rotation_repr}")
        print(f"  Equivariant embedding: {args.equivariant}")

        model = model_builders[args.model](
            rotation_repr=args.rotation_repr,
            use_equivariant_embedding=args.equivariant,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            rotation, translation = model(x)

        print(f"\nForward pass successful!")
        print(f"  Rotation output: {tuple(rotation.shape)}")
        print(f"  Translation output: {tuple(translation.shape)}")
