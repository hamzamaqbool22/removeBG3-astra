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

    def test_patterned_floor_keeps_shadow_continuous(self):
        rgb, alpha = self.scene()
        yy, xx = np.mgrid[:240,:400]
        tiles = np.where((xx//6 + yy//6)%2, 210., 165.)
        attenuation = .65*np.exp(-((xx-200)/100)**4-((yy-160)/28)**2)
        ground = tiles*(1-attenuation)
        rgb[:] = ground[...,None].astype(np.uint8)
        rgb[alpha>0] = 40
        shadow = recover_shadow(rgb,alpha)
        self.assertIsNotNone(shadow)
        # Across alternating tiles, the shadow must remain connected.
        self.assertGreater(float(shadow[170,150:250].min()), .20)
        self.assertLess(float(shadow[:30].max()), .01)
        self.assertLess(float(shadow[220:].max()), .08)

    def test_large_contrast_tiles_use_synthetic_fallback(self):
        rgb, alpha = self.scene()
        yy, xx = np.mgrid[:240,:400]
        # Large checkerboard patches survive a small blur. They must not be
        # mistaken for a strong photographed shadow and bypass completion.
        rgb[:] = np.where((xx//18 + yy//18)%2,220,125)[...,None]
        rgb[alpha>0] = 30
        self.assertIsNone(recover_shadow(rgb,alpha))

    def test_dark_car_on_patterned_floor_does_not_bleed(self):
        rgb, alpha = self.scene()
        yy, xx = np.mgrid[:240,:400]
        rgb[:] = np.where((xx//6 + yy//6)%2,210,165)[...,None]
        rgb[alpha>0] = 0
        shadow = recover_shadow(rgb,alpha)
        if shadow is not None:
            self.assertLess(float(shadow[alpha==0].max()), .16)

if __name__ == '__main__':
    unittest.main()
