import numpy as np
import pytest
from test.utils import convert_and_test, NP_SEED
from torchvision.models import shufflenet_v2_x0_5



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [shufflenet_v2_x0_5])
@pytest.mark.parametrize('pretrained', [True])
@pytest.mark.skip(reason="Does not export  Pytorch->Onnx well due to torch.chunck")
def test_shufflenet(pretrained, model_class):
    np.random.seed(seed=NP_SEED)
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
