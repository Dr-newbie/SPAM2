import os
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model, LoraConfig
from Foundations import inf_encoder_factory
from gene_encoder import GeneEncoder

class SPAM(nn.Module):
    def __init__(self, backbone, hidden_dim, proj_dim, dropout = 0.25, batch_norm=False, aux_output=250, embed_type=None):
        super(SPAM, self).__init__()

        self.embed_type = embed_type

        backbone_embed_dim_dict = {"hoptimus0": 1536, "gigapath": 1536, 
                                   "virchow": 2560, "virchow2": 2560, 
                                   "uni_v1": 1024, "conch_v1": 512, "plip": 768,
                                   "phikon": 768, "ctranspath": 768,
                                   "resnet50": 512}
        
        backbone_out_embed_dim = backbone_embed_dim_dict[backbone]

        self.projector_head = nn
