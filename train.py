import torch

from model import subsequent_mask

class Batch:
    def __init__(self,src,tgt=None,pad=2):
        self.src=src
        self.src_mask=(src!=pad).unsqueeze(-2)

    if tgt is not None:
        self.tgt=tgt[:,:-1]
        self.tgt_y=tgt[:,1:]