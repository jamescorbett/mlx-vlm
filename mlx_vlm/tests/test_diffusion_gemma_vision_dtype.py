"""float16 vision weights that cannot survive the forward pass.

The vision tower carries an input-normalization constant (``std_bias``) close to
54000. float16 tops out at 65504, so a repack that casts the tower from bfloat16
to float16 leaves no headroom: the normalization overflows to inf and every
image decodes to NaN. Observed on a real MXFP4 repack whose vision weights were
bit-identical to the bfloat16 reference apart from their dtype.
"""

import unittest

import mlx.core as mx

from mlx_vlm.models.diffusion_gemma.diffusion_gemma import (
    _promote_vision_weights,
    _vision_needs_promotion,
)

# The actual value in the affected checkpoint.
STD_BIAS = 53760.0


class TestVisionDtypePromotion(unittest.TestCase):
    def test_detects_a_weight_near_the_fp16_ceiling(self):
        weights = {"std_bias": mx.array([STD_BIAS], dtype=mx.float16)}
        self.assertTrue(_vision_needs_promotion(weights.values()))

    def test_ordinary_fp16_weights_are_left_alone(self):
        weights = {
            "layernorm.weight": mx.array([70.5], dtype=mx.float16),
            "q_proj.weight": mx.array([0.53], dtype=mx.float16),
        }
        self.assertFalse(_vision_needs_promotion(weights.values()))

    def test_bfloat16_checkpoints_are_never_touched(self):
        # The reference checkpoint holds the same value in bfloat16, whose range
        # is far wider, and must not be disturbed.
        weights = {"std_bias": mx.array([STD_BIAS], dtype=mx.bfloat16)}
        self.assertFalse(_vision_needs_promotion(weights.values()))

    def test_empty_weights_do_not_trip_detection(self):
        self.assertFalse(_vision_needs_promotion([]))
        self.assertFalse(
            _vision_needs_promotion([mx.array([], dtype=mx.float16)])
        )

    def test_promotion_moves_the_whole_tower(self):
        # Promoting only the offending tensor would leave float16 and bfloat16
        # mixed in one module, which fails where two of them meet in a matmul.
        weights = {
            "std_bias": mx.array([STD_BIAS], dtype=mx.float16),
            "q_proj.weight": mx.array([0.53], dtype=mx.float16),
        }
        promoted = _promote_vision_weights(weights)
        self.assertEqual({v.dtype for v in promoted.values()}, {mx.bfloat16})

    def test_promotion_preserves_the_values(self):
        weights = {"std_bias": mx.array([STD_BIAS], dtype=mx.float16)}
        promoted = _promote_vision_weights(weights)
        self.assertAlmostEqual(
            float(promoted["std_bias"][0]), STD_BIAS, delta=STD_BIAS * 0.01
        )

    def test_non_float_weights_pass_through(self):
        weights = {"ids": mx.array([1, 2, 3], dtype=mx.int32)}
        self.assertEqual(_promote_vision_weights(weights)["ids"].dtype, mx.int32)

    def test_fp16_survives_the_arithmetic_after_promotion(self):
        # The failure mode itself: squaring std_bias overflows in float16 and
        # does not in bfloat16.
        as_fp16 = mx.array([STD_BIAS], dtype=mx.float16)
        self.assertFalse(bool(mx.all(mx.isfinite(as_fp16 * as_fp16)).item()))
        promoted = _promote_vision_weights({"b": as_fp16})["b"]
        self.assertTrue(bool(mx.all(mx.isfinite(promoted * promoted)).item()))


if __name__ == "__main__":
    unittest.main()
