"""Run with .venv/bin/python local-test/check_contact_shadow.py."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vehicle_pipeline.contact_shadow import complete_contact_shadow

class ContactChecks(unittest.TestCase):
    def scene(self):
        a=np.zeros((240,400),np.uint8);a[40:160,80:320]=255
        return a
    def test_missing_shadow_gets_grounded_without_vertical_artifacts(self):
        a=self.scene();r,layer=complete_contact_shadow(a,None)
        self.assertGreater(float(r[160,200]),.8)
        self.assertEqual(float(r[:150].max()),0)
        self.assertEqual(float(r[:, :80].max()),0)
        self.assertEqual(float(r[210:].max()),0)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue(np.all(np.diff(r[160:190,200])<=1e-5))
        self.assertLessEqual(float(r.max()),.97)
    def test_strong_source_is_bit_exact(self):
        a=self.scene();s=np.zeros(a.shape,np.float32);s[160:190,80:320]=.7
        r,layer=complete_contact_shadow(a,s)
        np.testing.assert_array_equal(r,s)
        self.assertEqual(float(layer.max()),0)
    def test_narrow_dark_source_is_preserved(self):
        a=self.scene();s=np.zeros(a.shape,np.float32);s[160:164,80:320]=.90
        r,layer=complete_contact_shadow(a,s)
        np.testing.assert_array_equal(r,s)
        self.assertEqual(float(layer.max()),0)
    def test_weak_source_improves_without_stacking(self):
        a=self.scene();s=np.zeros(a.shape,np.float32);s[160:175,80:320]=.25
        saved=s.copy();original=a.copy();r,layer=complete_contact_shadow(a,s)
        self.assertGreater(float(r[160,200]),.8)
        np.testing.assert_array_equal(r,layer)
        np.testing.assert_array_equal(saved,s);np.testing.assert_array_equal(original,a)
    def test_weak_disconnected_fragments_are_replaced(self):
        a=self.scene();s=np.zeros(a.shape,np.float32);s[190:200,300:360]=.4
        r,_=complete_contact_shadow(a,s)
        self.assertEqual(float(r[:,320:].max()),0)
    def test_cropped_contacts_do_not_index_outside_image(self):
        a=self.scene();a[160:]=a[159]
        s=np.full(a.shape,.2,np.float32)
        r,_=complete_contact_shadow(a,s)
        np.testing.assert_array_equal(r,s)
    def test_front_footprint_has_visible_depth_and_soft_falloff(self):
        a=np.zeros((400,400),np.uint8);a[40:240,80:320]=255
        r,_=complete_contact_shadow(a,None)
        self.assertGreater(float(r[255,200]),.45)
        self.assertGreater(float(r[270,200]),.1)
        self.assertLess(float(r[290,200]),.05)
        self.assertTrue(np.all(np.diff(r[240:320,200])<=1e-5))
    def test_empty_mask(self):
        a=np.zeros((20,30),np.uint8);r,_=complete_contact_shadow(a,None)
        self.assertEqual(float(r.sum()),0)

if __name__=='__main__':unittest.main()
