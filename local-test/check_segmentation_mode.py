"""Test optional single-pass inference without loading a model."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_pipeline.segmentation import Segmenter

class FakeSession:
    def __init__(self, mask):
        self.mask = mask
        self.calls = []
    def predict(self, image):
        self.calls.append(image.size)
        if len(self.calls) == 1:
            return [Image.fromarray(self.mask)]
        return [Image.new('L', image.size, 255)]

class ModeChecks(unittest.TestCase):
    def segmenter(self, mode, large=False):
        mask = np.zeros((200, 300), np.uint8)
        if large:
            mask[20:180, 20:280] = 255
        else:
            mask[80:130, 110:190] = 255
        seg = Segmenter.__new__(Segmenter)
        seg.segmentation_mode = mode
        seg.session = FakeSession(mask)
        return seg, mask
    def test_single_preserves_first_mask_and_skips_second_call(self):
        seg, mask = self.segmenter('single')
        actual = seg.predict(np.zeros((200, 300, 3), np.uint8))
        np.testing.assert_array_equal(actual, mask)
        self.assertEqual(len(seg.session.calls), 1)
        self.assertEqual(seg.last_info, {'inference_passes': 1, 'adaptive_crop': None})
    def test_auto_still_refines_small_foreground(self):
        seg, _ = self.segmenter('auto')
        actual = seg.predict(np.zeros((200, 300, 3), np.uint8))
        self.assertEqual(actual.shape, (200, 300))
        self.assertEqual(len(seg.session.calls), 2)
        self.assertEqual(seg.last_info['inference_passes'], 2)
        self.assertIsNotNone(seg.last_info['adaptive_crop'])
    def test_large_foreground_unchanged_between_modes(self):
        for mode in ('auto', 'single'):
            seg, mask = self.segmenter(mode, large=True)
            np.testing.assert_array_equal(seg.predict(np.zeros((200, 300, 3), np.uint8)), mask)
            self.assertEqual(len(seg.session.calls), 1)
    def test_empty_foreground_has_no_second_pass(self):
        seg, mask = self.segmenter('auto')
        mask[:] = 0
        seg.predict(np.zeros((200, 300, 3), np.uint8))
        self.assertEqual(seg.last_info['inference_passes'], 1)
    def test_invalid_mode_fails_before_loading_model(self):
        with patch.dict(os.environ, {'SEGMENTATION_MODE': 'wrong'}):
            with self.assertRaisesRegex(ValueError, 'SEGMENTATION_MODE'):
                Segmenter()

if __name__ == '__main__':
    unittest.main()
