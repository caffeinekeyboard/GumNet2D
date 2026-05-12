import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

from GumNet2D.model.gumnet_non_linear_alignment import GumNetNonLinearAlignment

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def nonlinear_alignment():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP TESTS
# -----------------------------------------------------------------------------

def test_model_initialization(nonlinear_alignment):
    assert isinstance(nonlinear_alignment, GumNetNonLinearAlignment)
    assert hasattr(nonlinear_alignment, 'fc1')
    assert hasattr(nonlinear_alignment, 'fc2')
    assert hasattr(nonlinear_alignment, 'fc_out')
    assert hasattr(nonlinear_alignment, 'grid_size')
    assert hasattr(nonlinear_alignment, 'source_pts')
    assert nonlinear_alignment.source_pts.shape == (16, 2)
    assert nonlinear_alignment.source_pts.min() >= -0.8 - 1e-6
    assert nonlinear_alignment.source_pts.max() <= 0.8 + 1e-6


def test_identity_initialization(nonlinear_alignment):
    out_weight = nonlinear_alignment.fc_out.weight
    out_bias = nonlinear_alignment.fc_out.bias
    assert torch.all(out_weight == 0), "Output layer weights are not zero-initialized."
    assert torch.all(out_bias == 0), "Output layer biases are not zero-initialized."


def test_identity_warp_at_init():
    """Zero-initialized head must produce an identity warp regardless of input."""
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    model.eval()

    c_ab = torch.randn(2, 4096)
    c_ba = torch.randn(2, 4096)
    source = torch.randn(2, 1, 192, 192)

    with torch.no_grad():
        warped, target_pts, _, disp = model(c_ab, c_ba, source)

    # target_pts should coincide with source_pts (broadcast across batch).
    expected_target = model.source_pts.unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(target_pts, expected_target, atol=1e-6)

    # Displacement should be ~zero everywhere (a TPS solving target=source is the identity).
    assert disp.abs().max() < 1e-4, f"Identity warp produced disp={disp.abs().max().item()}"

    # And the warped image should match the source up to small bilinear/border effects.
    assert torch.allclose(warped, source, atol=1e-3)


# -----------------------------------------------------------------------------
# 2. FUNCTIONAL TESTS (DIMENSIONALITY & CORRECTNESS)
# -----------------------------------------------------------------------------

def test_forward_pass_dimensions(nonlinear_alignment):
    batch_size = 2
    feature_dim = 4096
    channels = 1
    spatial = 192
    grid_size = 4
    N = grid_size * grid_size

    c_ab = torch.randn(batch_size, feature_dim)
    c_ba = torch.randn(batch_size, feature_dim)
    source_image = torch.randn(batch_size, channels, spatial, spatial)

    with torch.no_grad():
        warped_image, target_pts, l_map, disp = nonlinear_alignment(c_ab, c_ba, source_image)

    assert warped_image.shape == (batch_size, channels, spatial, spatial)
    assert target_pts.shape == (batch_size, N, 2)
    assert l_map.shape == (batch_size, 1, spatial, spatial)
    assert disp.shape == (batch_size, 2, spatial, spatial)


def test_lambda_map_is_positive_and_finite(nonlinear_alignment):
    """l_map must be strictly positive and finite — it's a regularization weight."""
    source = torch.randn(2, 1, 192, 192)
    with torch.no_grad():
        l_map = nonlinear_alignment._compute_lambda_map(source)
    assert torch.isfinite(l_map).all()
    assert (l_map > 0).all()


def test_warp_is_plug_and_play_at_higher_resolution(nonlinear_alignment):
    """`warp(...)` should apply the same TPS to an image of any spatial size."""
    c_ab = torch.randn(1, 4096)
    c_ba = torch.randn(1, 4096)
    source_192 = torch.randn(1, 1, 192, 192)

    with torch.no_grad():
        _, target_pts, l_map, _ = nonlinear_alignment(c_ab, c_ba, source_192)
        # Simulate the full-res branch: same deformation, denser sampling.
        source_full = F.interpolate(source_192, size=(384, 384), mode='bilinear', align_corners=True)
        warped_full = nonlinear_alignment.warp(source_full, target_pts, l_map)

    assert warped_full.shape == (1, 1, 384, 384)
    assert torch.isfinite(warped_full).all()


# -----------------------------------------------------------------------------
# 3. TRAINING & GRAPH INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_backward_pass_and_differentiability():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    nn.init.normal_(model.fc_out.weight, std=0.01)

    model.train()

    c_ab = torch.randn(2, 4096, requires_grad=True)
    c_ba = torch.randn(2, 4096, requires_grad=True)
    source_image = torch.randn(2, 1, 192, 192, requires_grad=True)

    warped_image, target_pts, l_map, disp = model(c_ab, c_ba, source_image)
    dummy_loss = warped_image.sum() + target_pts.sum() + l_map.sum() + disp.sum()
    dummy_loss.backward()

    assert c_ab.grad is not None, "Gradients did not reach input c_ab."
    assert c_ba.grad is not None, "Gradients did not reach input c_ba."
    assert source_image.grad is not None, "Gradients did not reach input source_image."

    fc1_grad = model.fc1.weight.grad
    fc2_grad = model.fc2.weight.grad
    fc_out_grad = model.fc_out.weight.grad

    assert fc1_grad is not None and torch.sum(torch.abs(fc1_grad)) > 0, \
        "Gradients failed to update fc1."
    assert fc2_grad is not None and torch.sum(torch.abs(fc2_grad)) > 0, \
        "Gradients failed to update fc2."
    assert fc_out_grad is not None and torch.sum(torch.abs(fc_out_grad)) > 0, \
        "Gradients failed to update fc_out."


# -----------------------------------------------------------------------------
# 4. HARDWARE & ECOSYSTEM COMPATIBILITY TESTS
# -----------------------------------------------------------------------------

def test_device_agnosticism():
    model = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = model.to(device)
        c_ab = torch.randn(2, 4096).to(device)
        c_ba = torch.randn(2, 4096).to(device)
        source_image = torch.randn(2, 1, 192, 192).to(device)

        with torch.no_grad():
            warped_image, target_pts, l_map, disp = model(c_ab, c_ba, source_image)

        for t in (warped_image, target_pts, l_map, disp):
            assert t.device.type == device.type
            assert t.device == source_image.device

    except Exception as e:
        pytest.fail(f"Model failed on device {device}. Error: {e}")


def test_variable_batch_sizes(nonlinear_alignment):
    for B in (1, 4):
        c_ab = torch.randn(B, 4096)
        c_ba = torch.randn(B, 4096)
        img = torch.randn(B, 1, 192, 192)

        with torch.no_grad():
            warped, target_pts, l_map, disp = nonlinear_alignment(c_ab, c_ba, img)

        assert warped.shape == (B, 1, 192, 192)
        assert target_pts.shape == (B, 16, 2)
        assert l_map.shape == (B, 1, 192, 192)
        assert disp.shape == (B, 2, 192, 192)
