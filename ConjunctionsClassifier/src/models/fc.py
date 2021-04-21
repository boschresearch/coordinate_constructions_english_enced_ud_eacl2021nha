#  Copyright (c) 2021 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

import torch.nn as nn


class FC(nn.Module):
    """Fully connected layer with dropout."""
    def __init__(self, n_in, n_hidden, n_out, activation, p_dropout=0.0):
        super(FC, self).__init__()

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        assert isinstance(n_hidden, list)

        layer_dims = [n_in] + n_hidden + [n_out]
        self.layers = nn.ModuleList([nn.Linear(layer_dims[i-1], layer_dims[i]) for i in range(1, len(layer_dims))])

        self.activation = activation
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
            x = self.activation(x)

        return x

