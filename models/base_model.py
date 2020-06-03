import numpy as np

import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, valid_padding, \
                                        SlimConv2d, SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

class BaseModel(TorchModelV2, nn.Module):
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        layers = []

        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]

        # First conv layer
        out_channels = 6
        kernel = [3, 3]
        stride = 1
        padding, out_size = valid_padding(in_size, kernel, [stride, stride])

        self._conv = SlimConv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding
        )

        out_flatten_size = int(out_channels * out_size[0] * out_size[1])

        self._logits = SlimFC(
            out_flatten_size, num_outputs, initializer=nn.init.xavier_uniform_)
        self._value_branch = SlimFC(
            out_flatten_size, 1, initializer=normc_initializer())
        # Holds the current "base" output (before logits layer).
        self._features = None

    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._features = self._conv(input_dict["obs"].permute(0, 3, 1, 2).float())
        logits = self._logits(self._features.reshape(self._features.shape[0], -1))
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features.reshape(self._features.shape[0], -1)).squeeze(1)
        return value
