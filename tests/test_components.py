import einops
import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.components import LinearComponent


def reference_forward_with_mask(
    x: Float[Tensor, "batch ... d_in"],
    A: Float[Tensor, "... C d_in m"],
    B: Float[Tensor, "... C m d_out"],
    topk_mask: Float[Tensor, "batch ... C"],
) -> Float[Tensor, "batch ... d_out"]:
    """Reference implementation that applies the mask after the full computation, rather than
    after the first multiplication by A (which is done for efficiency in the code).
    """
    # Apply A and B matrices
    inner = einops.einsum(x, A, "batch ... d_in, ... C d_in m -> batch ... C m")
    comp_acts = einops.einsum(inner, B, "batch ... C m, ... C m d_out -> batch ... C d_out")

    # Apply mask and sum
    out = einops.einsum(comp_acts, topk_mask, "batch ... C d_out, batch ... C -> batch ... d_out")

    return out


def test_linear_component_mask_values():
    """Test that masking works correctly with different mask values."""
    batch_size, d_in, d_out, C, m = 2, 8, 8, 4, 4

    component = LinearComponent(d_in=d_in, d_out=d_out, C=C, m=m)
    x = torch.randn(batch_size, d_in)

    # Test with various mask patterns
    test_masks = [
        torch.ones(batch_size, C),  # All ones
        torch.zeros(batch_size, C),  # All zeros
        torch.eye(C)[None].expand(batch_size, -1, -1)[:, :C],  # Identity-like
        torch.rand(batch_size, C),  # Random values
    ]

    for topk_mask in test_masks:
        actual_output = component(x, topk_mask)
        expected_output = reference_forward_with_mask(x, component.A, component.B, topk_mask)
        torch.testing.assert_close(actual_output, expected_output)
