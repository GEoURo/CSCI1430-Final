import torch
import torch.nn as nn


def getEncoder(
        num_freq: int = 3,
        log_scale: bool = True,
        i: int = 0
):
    if i == -1:
        return nn.Identity(), 3

    encoder = PositionalEncoding(num_freq, log_scale=log_scale)

    def encode(x):
        return encoder.encode(x)

    return encode, encoder.getOutputChannels()


class PositionalEncoding:
    """
    Implement the forward pass of the positional encoding, which encode x to (x, sin(2^k * x), cos(2^k * x), ...)
    Args:
        input_channels: the number of channels of the input (3 for xyz and direction)
        num_freq: `L_d=4` for viewing direction, `L_x=10` for 3D coordinates
        log_scale: whether to use log scale for frequency bands,
            `log_scale=True` for generating 0-9 first, and take the power of 2 separately
            `log_scale=False` for taking the power of 2 first, and split equally

    """

    def __init__(self, num_freq, input_channels=3, log_scale=True):
        super(PositionalEncoding, self).__init__()

        self.num_freq = num_freq
        self.input_channels = input_channels
        self.encode_func = [torch.sin, torch.cos]

        self.output_channels = input_channels * (len(self.encode_func) * num_freq + 1)
        if log_scale:
            self.freq_bands = 2 ** torch.linspace(0, num_freq - 1, num_freq)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freq - 1), num_freq)

    def encode(self, x):
        """
        Inputs:
            x: (ray_cnt,num_samples, input_channels)
        Outputs:
            x: (ray_cnt,num_samples, output_channels)
        """
        output = [x]
        for f in self.freq_bands:
            for func in self.encode_func:
                output.append(func(f * x))
        return torch.cat(output, dim=-1)

    def getOutputChannels(self):
        return self.output_channels

    def getInputChannels(self):
        return self.input_channels
