import torch
from torch import nn

class Persistence(nn.Module):
    def __init__(self, layout="NTHWC"):
        super(Persistence, self).__init__()
        self.layout = layout
        self.t_axis = self.layout.find("T")

    def forward(self, in_seq, out_seq):
        last_frame = in_seq.select(self.t_axis, -1)
        out_len = out_seq.shape[self.t_axis]
        repeated = last_frame.unsqueeze(self.t_axis).repeat_interleave(out_len, dim=self.t_axis)
        return repeated, out_seq