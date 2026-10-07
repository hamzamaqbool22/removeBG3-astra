"""CPU checks for source preservation, without loading a segmentation model."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_first import choose_shadow, expand_ground_shadow, deepen_weak_footprint, finish_ground_shadow


class SourceFirstChecks(unittest.TestCase):
    def test_rounding_reduces_square_corner_without_changing_car(self):
        alpha=self.scene()
        shadow=np.zeros(alpha.shape,np.float32)
        shadow[160:205,40:360]=.7
        result=finish_ground_shadow(alpha,shadow)
        self.assertLess(float(result[204,40]),float(shadow[204,40]))
        np.testing.assert_array_equal(result[alpha>127],shadow[alpha>127])
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(result.max()),.970001)

    def test_deeper_footprint_preserves_contact_and_fades_at_canvas_edge(self):
        alpha=self.scene()
        yy=np.arange(240)[:,None]
        footprint=np.zeros((240,400),np.float32)
        footprint[:,80:320]=.8*np.exp(-.5*((yy-159)/12)**2)
        footprint[alpha>127]=0
        result=deepen_weak_footprint(alpha,footprint)
        self.assertGreater(float(result[185,200]),float(footprint[185,200])+.1)
        self.assertTrue((result>=footprint).all())
        np.testing.assert_array_equal(result[-1],footprint[-1])
        self.assertEqual(float(result[alpha>127].max()),0)

    def scene(self):
        alpha=np.zeros((240,400),np.uint8)
        alpha[40:160,60:340]=255
        return alpha

    def test_asymmetric_real_shadow_is_not_replaced(self):
        alpha=self.scene()
        source=np.zeros(alpha.shape,np.float32)
        source[160:180,170:340]=.75
        result,mode=choose_shadow(alpha,source)
        self.assertEqual(mode,'preserved-source-with-contact-fill')
        np.testing.assert_array_equal(result[165:180,170:340],source[165:180,170:340])
        self.assertLess(float(result[178,100]),.01)

    def test_single_patch_does_not_count_as_full_source(self):
        source=np.zeros((240,400),np.float32);source[160:170,170:180]=.8
        _,mode=choose_shadow(self.scene(),source)
        self.assertEqual(mode,'synthetic-fallback')

    def test_thin_distributed_source_gets_ground_completion(self):
        alpha=self.scene()
        source=np.zeros(alpha.shape,np.float32)
        source[160:165,85:315]=.65
        result,mode=choose_shadow(alpha,source)
        self.assertEqual(mode,'source-with-weak-shadow-completion')
        self.assertGreater(float(result[170,200]),.1)
        self.assertTrue((result>=source).all())

    def test_empty_and_missing_are_finite(self):
        for alpha in (np.zeros((240,400),np.uint8),self.scene()):
            result,_=choose_shadow(alpha,None)
            self.assertTrue(np.isfinite(result).all())
            self.assertGreaterEqual(float(result.min()),0)
            self.assertLessEqual(float(result.max()),.97)

    def test_expansion_preserves_core_and_stays_on_ground(self):
        alpha=self.scene()
        source=np.zeros(alpha.shape,np.float32)
        source[160:180,100:300]=.8
        saved=source.copy()
        result=expand_ground_shadow(alpha,source)
        np.testing.assert_array_equal(result[165:175,110:290],source[165:175,110:290])
        self.assertGreater(float(result[170,98]),.1)
        self.assertGreater(float(result[180,200]),.1)
        self.assertEqual(float(result[:160].max()),0)
        self.assertLess(float(result[200:].max()),.001)
        np.testing.assert_array_equal(source,saved)
        np.testing.assert_array_equal(expand_ground_shadow(alpha,source,0),source)

    def test_hidden_opacity_cannot_generate_halo(self):
        alpha=self.scene()
        source=(alpha>127).astype(np.float32)*.9
        result=expand_ground_shadow(alpha,source)
        np.testing.assert_array_equal(result,source)


if __name__=='__main__':unittest.main()
