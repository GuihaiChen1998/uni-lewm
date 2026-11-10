import pytest
import torch

from fused_lewm.model import ModelConfig, WorldModel


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(2)
    torch.manual_seed(11)


@pytest.fixture
def model():
    return WorldModel(
        ModelConfig(
            image_size=16,
            patch_size=8,
            dim=24,
            depth=2,
            heads=3,
            mlp_ratio=2,
            history=3,
            projection_batchnorm=True,
        )
    )
