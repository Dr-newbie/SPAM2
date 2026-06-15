import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both sys.path layouts (Code_final/code or Code_final).
try:
    from utils.attention import CrossAttentionBlock
except ModuleNotFoundError:
    from code.utils.attention import CrossAttentionBlock

class CrossModalFusion(nn.Module):
    """
    입력:
      z_img  : (B, Di)  — 이미지 임베딩(패치/풀링 후 1토큰이라고 가정)
      z_spot : (B, Ds)  — 공간(GCN) 임베딩(1토큰)
      z_gene : (B, Dg)  — 유전자 임베딩(1토큰; SNN/MLP 인코더 출력)
    처리:
      img<-spot cross-attn, img<-gene cross-attn → 두 토큰을 게이트합/concat+proj
    출력:
      z_fused: (B, D)
    """
    def __init__(self, dim_img: int, dim_spot: int, dim_gene: int,
                 dim_out: int, num_heads: int = 8, dropout: float = 0.1, ffn_hidden: int = 4096,
                 merge: str = "gated-sum",
                 img_tokens: int = 1, spot_tokens: int = 1, gene_tokens: int = 1):  # "gated-sum" | "concat-proj"
        super().__init__()
        if img_tokens < 1 or spot_tokens < 1 or gene_tokens < 1:
            raise ValueError("img_tokens/spot_tokens/gene_tokens must be >= 1")
        self.merge = merge
        self.img_tokens = int(img_tokens)
        self.spot_tokens = int(spot_tokens)
        self.gene_tokens = int(gene_tokens)
        self.dim_img = dim_img
        self.dim_spot = dim_spot
        self.dim_gene = dim_gene
        self.xa_is = CrossAttentionBlock(dim_q=dim_img, dim_kv=dim_spot, dim_out=dim_out,
                                         num_heads=num_heads, dropout=dropout, ffn_hidden=ffn_hidden)
        self.xa_ig = CrossAttentionBlock(dim_q=dim_img, dim_kv=dim_gene, dim_out=dim_out,
                                         num_heads=num_heads, dropout=dropout, ffn_hidden=ffn_hidden)

        self.img_tokenizer = None if self.img_tokens == 1 else nn.Linear(dim_img, dim_img * self.img_tokens)
        self.spot_tokenizer = None if self.spot_tokens == 1 else nn.Linear(dim_spot, dim_spot * self.spot_tokens)
        self.gene_tokenizer = None if self.gene_tokens == 1 else nn.Linear(dim_gene, dim_gene * self.gene_tokens)

        if merge == "gated-sum":
            self.gate = nn.Sequential(
                nn.Linear(dim_out * 2, dim_out),
                nn.Sigmoid()
            )
        elif merge == "concat-proj":
            self.proj = nn.Linear(dim_out * 2, dim_out)
        else:
            raise ValueError(f"Unknown merge: {merge}")

    def _to_tokens(self, x, tokenizer, n_tokens: int, token_dim: int):
        if n_tokens == 1:
            return x.unsqueeze(1)
        t = tokenizer(x)
        return t.view(x.size(0), n_tokens, token_dim)

    def _pool_tokens(self, x):
        if x.size(1) == 1:
            return x.squeeze(1)
        return x.mean(dim=1)

    def forward(self, z_img, z_spot, z_gene, return_attn: bool = False):
        Qi = self._to_tokens(z_img, self.img_tokenizer, self.img_tokens, self.dim_img)
        Ks = self._to_tokens(z_spot, self.spot_tokenizer, self.spot_tokens, self.dim_spot)
        Kg = self._to_tokens(z_gene, self.gene_tokenizer, self.gene_tokens, self.dim_gene)

        # img<-spot / img<-gene
        if return_attn:
            his, attn_is = self.xa_is(Qi, Ks, Ks, return_attn=True)
            hig, attn_ig = self.xa_ig(Qi, Kg, Kg, return_attn=True)
        else:
            his = self.xa_is(Qi, Ks, Ks)  # (B,T,Dout)
            hig = self.xa_ig(Qi, Kg, Kg)  # (B,T,Dout)
            attn_is, attn_ig = None, None

        # merge
        if self.merge == "gated-sum":
            cat = torch.cat([his, hig], dim=-1)     # (B,1,2D)
            gate = self.gate(cat)                   # (B,1,D)
            z = gate * his + (1.0 - gate) * hig     # (B,1,D)
        else:
            z = torch.cat([his, hig], dim=-1)       # (B,1,2D)
            z = self.proj(z)                        # (B,1,D)

        z_pooled = self._pool_tokens(z)
        his_pooled = self._pool_tokens(his)
        hig_pooled = self._pool_tokens(hig)
        if return_attn:
            return z_pooled, his_pooled, hig_pooled, {"attn_is": attn_is, "attn_ig": attn_ig}
        return z_pooled, his_pooled, hig_pooled
