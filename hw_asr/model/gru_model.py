from torch import nn

from hw_asr.base import BaseModel


class GRUModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=256, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.encoder = nn.GRU(n_feats, fc_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * fc_hidden, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.encoder(spectrogram.transpose(1, 2))
        output = self.linear(output)
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here