import torch
import torch.nn.functional as F

from gtv.probes.sets import clutter_scene
from gtv.thalamus import route_window, saliency_map, select_location, template_match_map


def test_route_window_recovers_the_item_object_centered():
    """Ground truth: an item placed at a known position in a scene is recovered
    by routing to its center: the routed image equals the item upsampled 2x and
    centered, as the stack's training images are."""
    g = torch.Generator().manual_seed(0)
    item = torch.zeros(28, 28)
    item[6:22, 10:18] = 1.0
    scene, centers = clutter_scene([item], 64, g, return_positions=True)
    routed = route_window(scene.unsqueeze(0), torch.tensor([centers[0]]))
    up = F.interpolate(item.view(1, 1, 28, 28), scale_factor=2, mode="bilinear", align_corners=False)
    ref = F.pad(up, (4, 4, 4, 4))
    ref = ref - ref.mean()
    assert routed.shape == (1, 1, 64, 64)
    assert torch.allclose(routed, ref, atol=1e-5)


def test_select_location_and_priority_maps():
    pr = torch.zeros(2, 8, 8)
    pr[0, 2, 5], pr[1, 7, 0] = 3.0, 1.0
    c = select_location(pr, 64)
    assert torch.equal(c, torch.tensor([[20.0, 44.0], [60.0, 4.0]]))
    templates = torch.zeros(3, 4)
    templates[1, 2] = 1.0
    f = torch.zeros(1, 4, 8, 8)
    f[0, 2, 3, 3] = 2.0  # the target's feature at (3, 3)
    f[0, 0, 6, 6] = 5.0  # something else, stronger, at (6, 6)
    assert int(template_match_map(f, templates, 1).flatten().argmax()) == 3 * 8 + 3
    assert int(saliency_map(f).flatten().argmax()) == 6 * 8 + 6


def test_topk_glimpses_suppress_neighbors_and_border():
    from gtv.thalamus import learned_priority, select_topk

    pr = torch.zeros(1, 8, 8)
    pr[0, 0, 0] = 9.0  # border artifact
    pr[0, 3, 3], pr[0, 3, 4], pr[0, 6, 2] = 5.0, 4.9, 3.0  # object at (3, 3), its neighbor, a second object
    c = select_topk(pr, 64, 2, border=1)
    assert torch.equal(c[0], torch.tensor([[28.0, 28.0], [52.0, 20.0]]))  # skips border and the neighbor
    f = torch.randn(2, 4, 8, 8)
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    lp = learned_priority(f, w, 0.5, torch.zeros(4), torch.ones(4))
    assert torch.allclose(lp, f[:, 0] + 0.5)
