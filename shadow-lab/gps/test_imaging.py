import unittest
import numpy as np
from imaging import apply_shadow, geometry_region, letterbox, restore_map, shadow_opacity


class ImagingTests(unittest.TestCase):
    def test_opaque_car_and_unshadowed_road_are_identical(self):
        rng = np.random.default_rng(1)
        bg = rng.integers(0,256,(80,120,3),dtype=np.uint8)
        alpha = np.zeros((80,120),np.uint8)
        alpha[20:60,30:90] = 255
        composite = bg.copy()
        composite[alpha==255] = [30,50,100]
        opacity = np.zeros(alpha.shape,np.float32)
        opacity[40:70,20:100] = .6
        out = apply_shadow(composite,bg,alpha,opacity)
        np.testing.assert_array_equal(out[alpha==255],composite[alpha==255])
        np.testing.assert_array_equal(out[opacity==0],composite[opacity==0])
        self.assertLess(out[65,50].sum(),bg[65,50].sum())

    def test_soft_edge_darkens_only_background_contribution(self):
        bg = np.full((1,1,3),200,np.uint8)
        alpha = np.full((1,1),128,np.uint8)
        car = np.full((1,1,3),40,np.float32)
        a = alpha[...,None]/255
        composite = np.uint8(np.rint(car*a+bg*(1-a)))
        out = apply_shadow(composite,bg,alpha,np.full((1,1),.5,np.float32))
        expected = np.uint8(np.rint(composite-bg.astype(float)*(1-a)*.5))
        np.testing.assert_array_equal(out,expected)

    def test_letterbox_restores_alignment_for_both_orientations(self):
        for h,w in [(768,1024),(1024,768)]:
            rgb = np.full((h,w,3),100,np.uint8)
            mask = np.zeros((h,w),np.uint8)
            mask[h//4:3*h//4,w//4:3*w//4] = 255
            _,small,box = letterbox(rgb,mask)
            restored = restore_map(small,box,mask.shape)>127
            self.assertGreater((restored & (mask>0)).sum()/(restored | (mask>0)).sum(),.99)
            self.assertEqual(small.shape,(512,512))

    def test_no_brightening_or_shadow_outside_support(self):
        before = np.full((32,32,3),100,np.uint8)
        support = np.zeros((32,32),np.float32)
        support[8:24,8:24] = 1
        bright = np.full_like(before,150)
        self.assertEqual(float(shadow_opacity(before,bright,support).max()),0)
        dark = np.full_like(before,50)
        opacity = shadow_opacity(before,dark,support)
        self.assertTrue(np.all(opacity[support==0]==0))
        self.assertAlmostEqual(float(opacity[16,16]),.5,places=5)

    def test_geometry_region_is_nonempty_and_does_not_mutate_input(self):
        prediction = np.array([0,0,0,0,.2],np.float32)
        before = prediction.copy()
        region = geometry_region(prediction,[256,350,160,60,-20])
        self.assertGreater(region.sum(),1000)
        self.assertLess(region.sum(),20000)
        np.testing.assert_array_equal(prediction,before)

    def test_rejects_bad_masks_and_geometry(self):
        with self.assertRaises(ValueError):
            letterbox(np.zeros((10,20,3),np.uint8),np.zeros((10,20),np.uint8))
        with self.assertRaises(ValueError):
            geometry_region([0,0,float('nan'),0,0],[1,1,2,2,0])


if __name__ == '__main__':
    unittest.main()
