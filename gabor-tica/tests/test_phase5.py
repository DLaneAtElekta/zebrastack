import pytest
import torch

from gtv.data import CLASSES, SUPERORDINATE
from gtv.data.categories import CACHE, load_fashion_mnist
from gtv.stages import HigherStage


def test_superordinate_groups_partition_the_classes():
    members = sorted(c for g in SUPERORDINATE.values() for c in g)
    assert members == list(range(len(CLASSES)))


@pytest.mark.skipif(not (CACHE / "t10k-images-idx3-ubyte.gz").exists(), reason="Fashion-MNIST not downloaded")
def test_fashion_mnist_loader_is_balanced_and_centered():
    x, y = load_fashion_mnist("test", 100, size=64)
    assert x.shape == (100, 1, 64, 64)
    assert torch.bincount(y, minlength=10).tolist() == [10] * 10
    assert torch.allclose(x.mean((2, 3)), torch.zeros(100, 1), atol=1e-5)
    assert x[:, :, :3].abs().max() < x.abs().max()  # content sits in the center


def test_higher_stage_shapes_skips_and_fit():
    torch.manual_seed(0)
    st = HigherStage("V4", in_channels=8, sheet=4, radius=1, skip_channels=3)
    x, skip = torch.randn(40, 8, 8, 8), torch.randn(40, 3, 16, 16)
    f = st.features(x, [skip])
    assert f.shape == (40, 8 + 3 + 8 * 4, 4, 4)
    st.fit(x, [skip], n_iter=30, polish_iter=5)
    s = st(x, [skip])
    assert s.shape == (40, 16, 4, 4)
    vec = f[3, :, 1, 2][None]
    assert torch.allclose(s[3, :, 1, 2], st.tica(vec)[0], atol=1e-3)  # forward matches the fitted TICA
    with pytest.raises(ValueError):
        HigherStage("big", in_channels=2, sheet=10)


def test_first_order_budget_all_keeps_the_pass_through():
    torch.manual_seed(0)
    st = HigherStage("V4", in_channels=6, sheet=4, radius=1, first_budget="all")  # 6 first + 10 second
    x = torch.randn(60, 6, 8, 8)
    st.fit(x, n_iter=20, polish_iter=5)
    assert st.tica.whitener.parts[0].components.shape[0] == 6
    # the stage's outputs determine its pooled inputs exactly (nothing of the pass-through is discarded)
    import torch.nn.functional as F

    s = st(x)
    first_hat = st.features_from_outputs(s)[:, :6]
    assert torch.allclose(first_hat, F.adaptive_avg_pool2d(x, 4), atol=1e-3)
    with pytest.raises(ValueError):
        HigherStage("V4", in_channels=16, sheet=4, first_budget="all")  # no room for second order
