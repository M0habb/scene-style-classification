import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits image into fixed-size patches and linearly projects each
    flattened patch into an embedding vector.

    A Conv2d with kernel_size=patch_size and stride=patch_size is
    mathematically identical to: flatten each patch → multiply by weight matrix.
    It's just faster.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.n_patches = (img_size // patch_size) ** 2  # e.g. (224//16)^2 = 196

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                   # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                   # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)             # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    One encoder block:
      x = x + MultiHeadAttention(LayerNorm(x))   ← attention sub-block
      x = x + MLP(LayerNorm(x))                  ← feed-forward sub-block

    The x + ... part is the residual (skip) connection.
    LayerNorm comes BEFORE each sub-block (pre-norm style).
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.2):
        super().__init__()

        # Attention sub-block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward sub-block
        self.norm2  = nn.LayerNorm(embed_dim)
        mlp_hidden  = int(embed_dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention with residual
        normed      = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)  # Q=K=V (self-attention)
        x           = x + self.attn_drop(attn_out)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) from scratch.

    Sequence of operations:
      1. Split image into patches, project each to embed_dim
      2. Prepend a learnable [CLS] token
      3. Add learnable positional embeddings to all tokens
      4. Pass through `depth` Transformer blocks
      5. Take only the [CLS] token output → classify
    """
    def __init__(
        self,
        img_size    = 224,
        patch_size  = 16,
        in_channels = 3,
        num_classes = 17,
        embed_dim   = 192,   # ViT-Tiny width
        depth       = 12,    # number of transformer blocks
        num_heads   = 3,     # must divide embed_dim evenly
        mlp_ratio   = 4.0,
        dropout     = 0.2,
    ):
        super().__init__()

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches        = self.patch_embed.n_patches

        # --- CLS token: one learnable vector prepended to the patch sequence ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional embedding: one vector per token (patches + CLS) ---
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # --- Transformer encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # --- Classification head ---
        self.head = nn.Linear(embed_dim, num_classes)

        # --- Weight initialisation ---
        self._init_weights()

    def _init_weights(self):
        # Standard ViT initialisation
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch embedding → (B, n_patches, embed_dim)
        x = self.patch_embed(x)

        # 2. Expand CLS token to batch size and prepend
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)           # (B, n_patches+1, embed_dim)

        # 3. Add positional embeddings (same shape as x)
        x = self.pos_drop(x + self.pos_embed)

        # 4. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 5. Take only CLS token → classify
        cls_output = x[:, 0]                       # (B, embed_dim)
        return self.head(cls_output)               # (B, num_classes)