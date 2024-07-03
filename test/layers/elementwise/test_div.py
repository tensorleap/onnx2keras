import numpy as np
import pytest
import torch.nn as nn

from test.utils import convert_and_test


class FDiv(nn.Module):
    def __init__(self):
        super(FDiv, self).__init__()

    def forward(self, x, y):
        x = x / 2
        y = y / 2

        x = x / y
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [False])
def test_div(change_ordering):
    model = FDiv()
    model.eval()

    input_np1 = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_np2 = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, (input_np1, input_np2), verbose=False, change_ordering=change_ordering)

