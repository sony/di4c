# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn
import torch.nn.functional as F
import math

import time

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn

class MaskTransformer(nn.Module):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000,
                 is_student=False):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
                latent_dim     -> int:     Dimension of latent (lambda in the paper) # added for Di4C
        """

        super().__init__()
        self.nclass = nclass
        self.patch_size = img_size // 16
        self.codebook_size = codebook_size
        self.tok_emb = nn.Embedding(codebook_size+1+nclass+1, hidden_dim)  # +1 for the mask of the viz token, +1 for mask of the class
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.patch_size*self.patch_size)+1, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(torch.zeros((self.patch_size*self.patch_size)+1, codebook_size+1+nclass+1))

        self.is_student = is_student
        if is_student:
            # Added for Di4C:
            self.latent_feature_dim = 128
            self.latent_projection = nn.Sequential(
                nn.Linear(in_features=self.latent_feature_dim, out_features=self.latent_feature_dim*4),
                nn.GELU(),
                nn.Linear(in_features=self.latent_feature_dim*4, out_features=hidden_dim)
            )
            # nn.init.zeros_(self.latent_projection[-1].weight)
            # nn.init.zeros_(self.latent_projection[-1].bias)

    def forward(self, img_token, y=None, drop_label=None, return_attn=False, z=None):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
                z              -> torch.FloatTensor: latent vector for mixrture modeling # added for Di4C
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h = img_token.size()

        cls_token = y.view(b, -1) + self.codebook_size + 1  # Shift the class token by the amount of codebook

        cls_token[drop_label] = self.codebook_size + 1 + self.nclass  # Drop condition
        input = torch.cat([img_token.view(b, -1), cls_token.view(b, -1)], -1)  # concat visual tokens and class tokens
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        if z is not None: # Di4C
            z_emb = transformer_timestep_embedding(
                z.view(-1) * 1000, self.latent_feature_dim
            )
            z_emb = self.latent_projection(z_emb)
            x = torch.cat([x, z_emb[:,None,:]], dim=1)

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        if z is not None: # Di4C
            x = x[:, :-1, :] # to ignore latent dimension

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.patch_size * self.patch_size, :self.codebook_size + 1], attn

        return logit[:, :self.patch_size*self.patch_size, :self.codebook_size+1]
    
# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb