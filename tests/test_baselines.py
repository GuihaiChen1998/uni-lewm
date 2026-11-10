import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from fused_lewm._vendor.lewm import SIGReg as UpstreamSIGReg
from fused_lewm.baselines import ToyBaseline
from fused_lewm.losses import SIGReg


def test_sigreg_coefficient_matches_upstream_normalization():
    torch.manual_seed(912)
    z = torch.randn(16, 4, 48)
    torch.manual_seed(81)
    upstream = UpstreamSIGReg(num_proj=32)(z.transpose(0, 1))
    torch.manual_seed(81)
    local = SIGReg(32)(z)
    torch.testing.assert_close(local, upstream / math.sqrt(2 * math.pi))


@pytest.mark.parametrize("method", ["lewm", "sub_lewm", "vlwm", "flexiworld"])
def test_named_models_train_both_parts_and_rollout_without_images(method):
    torch.manual_seed(42)
    model = ToyBaseline(method)
    obs = torch.rand(4, 6, 3, 32, 32)
    actions = torch.randint(4, (4, 5))
    loss, _ = model.loss(obs, actions, 0.9)
    loss.backward()
    for module in (model.encoder, model.predictor):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    model.eval()
    with torch.no_grad():
        history = model.encode_sequence(obs[:, :2])

        def fail_on_images(*_):
            raise AssertionError("latent rollout must never encode an image")

        hook = model.encoder.register_forward_pre_hook(fail_on_images)
        trajectory = model.rollout(history, actions[:, :1], actions[:, 1:])
        shorter = model.rollout(history, actions[:, :1], actions[:, 1:3])
        hook.remove()
    assert trajectory.shape == (4, 4, 48)
    torch.testing.assert_close(trajectory[:, :2], shorter)


def test_lewm_and_sub_lewm_start_with_identical_learned_weights():
    torch.manual_seed(7)
    lewm = ToyBaseline("lewm")
    torch.manual_seed(7)
    sub = ToyBaseline("sub_lewm")
    assert dict(lewm.named_parameters()).keys() == dict(sub.named_parameters()).keys()
    for name, value in lewm.named_parameters():
        torch.testing.assert_close(value, dict(sub.named_parameters())[name], rtol=0, atol=0)


def test_dreamer_replay_alignment_and_episode_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_dreamer_tf import prepare_batch
    from dreamer_data import GoalData

    data = GoalData(4)
    batch = data.next()
    assert set(data.train_ids).isdisjoint(data.validation_ids)
    assert len(data.train_ids) == 32 and len(data.validation_ids) == 8
    shifted = prepare_batch(batch)
    assert np.all(shifted["action"][:, 0] == 0)
    np.testing.assert_array_equal(shifted["action"][:, 1:].argmax(-1), batch["action"][:, :-1])
    np.testing.assert_array_equal(shifted["image"][..., :3], batch["image"])
    np.testing.assert_array_equal(shifted["image"][..., 3:], batch["goal"])
    observed_equal = np.all(batch["image"] == batch["goal"], axis=(2, 3, 4))
    np.testing.assert_array_equal(batch["reward"], observed_equal.astype(np.float32))
    assert np.all(batch["goal"] == batch["goal"][:, :1])
    assert np.all(batch["is_first"][:, 0]) and not np.any(batch["is_first"][:, 1:])
