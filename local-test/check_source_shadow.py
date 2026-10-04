"""Regression checks for the float shadow path that failed on the server."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_pipeline.source_shadow import recover_shadow

class ShadowRecoveryChecks(unittest.TestCase):
    def scene(self):
        rgb = np.full((240,400,3),180,np.uint8)
        alpha = np.zeros((240,400),np.uint8)
        alpha[50:160,90:310] = 255
        rgb[50:160,90:310] = 80
        return rgb, alpha

    def test_float_shadow_path_preserves_attached_shadow(self):
        rgb, alpha = self.scene()
        cv2.ellipse(rgb,(200,170),(100,20),0,0,360,(75,75,75),-1)
        shadow = recover_shadow(rgb,alpha)
        self.assertIsNotNone(shadow)
        self.assertEqual(shadow.shape,alpha.shape)
        self.assertTrue(np.isfinite(shadow).all())
        self.assertGreater(float(shadow[170,200]),.4)
        self.assertLess(float(shadow[:30].max()),.01)
        self.assertGreaterEqual(float(shadow.min()),0)
        self.assertLessEqual(float(shadow.max()),1)

    def test_floor_line_crossing_shadow_is_removed(self):
        rgb, alpha = self.scene()
        cv2.ellipse(rgb,(200,170),(100,20),0,0,360,(75,75,75),-1)
        # A thin dark seam connects to the real shadow and extends sideways.
        cv2.line(rgb,(45,178),(355,178),(15,15,15),3)
        shadow = recover_shadow(rgb,alpha)
        self.assertIsNotNone(shadow)
        self.assertLess(float(shadow[178,70]),.03)
        self.assertGreater(float(shadow[170,200]),.4)

    def test_flat_ground_does_not_invent_source_shadow(self):
        rgb, alpha = self.scene()
        self.assertIsNone(recover_shadow(rgb,alpha))

    def test_empty_mask_uses_fallback(self):
        rgb, alpha = self.scene()
        self.assertIsNone(recover_shadow(rgb,np.zeros_like(alpha)))

if __name__ == '__main__':
    unittest.main()
