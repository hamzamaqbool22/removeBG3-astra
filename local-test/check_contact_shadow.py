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
        self.assertGreater(float(r[160,200]),.7)
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
        self.assertGreater(float(r[160,200]),.7)
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
    def test_perspective_rear_contact_has_no_height_cutoff(self):
        # Rear contact sits higher in the picture than the nearest front tire.
        # The former 70%-height gate abruptly removed the right half of this shadow.
        a=np.zeros((300,500),np.uint8)
        for x in range(60,440):
            bottom=round(250-.30*(x-60))
            a[30:bottom+1,x]=255
        r,_=complete_contact_shadow(a,None)
        strengths=np.array([r[round(250-.30*(x-60))+4,x] for x in range(120,410)])
        self.assertGreater(float(strengths.min()),.20)
        self.assertLess(float(np.abs(np.diff(strengths)).max()),.08)
        self.assertEqual(float(r[a>127].max()),0)

    def test_empty_mask(self):
        a=np.zeros((20,30),np.uint8);r,_=complete_contact_shadow(a,None)
        self.assertEqual(float(r.sum()),0)

    def test_raised_underbody_is_grounded_between_tires(self):
        # Different wheelbases and clearances cover sedan/SUV/truck proportions.
        # The ground must remain dark beneath the raised body, not just tires.
        import cv2
        for clearance in (12, 35, 55):
            with self.subTest(clearance=clearance):
                a=np.zeros((320,540),np.uint8)
                a[40:230-clearance,50:490]=255
                cv2.ellipse(a,(135,190),(38,40),0,0,360,255,-1)
                cv2.ellipse(a,(405,190),(38,40),0,0,360,255,-1)
                r,_=complete_contact_shadow(a,None)
                self.assertGreater(float(r[225,200:340].min()),.45)
                self.assertEqual(float(r[a>127].max()),0)
                # A broad penumbra is intentional; it should fade away beyond
                # the footprint, not be clipped at the previous narrow radius.
                self.assertLess(float(r[300:].max()),.01)
                self.assertGreater(float(r[:,49].max()),.03)
                self.assertLess(float(np.abs(r[230:,50]-r[230:,49]).max()),.02)
                self.assertLess(float(r[:140].max()),.001)

    def test_three_perspective_contacts_and_mirror(self):
        import cv2
        a=np.zeros((320,540),np.uint8)
        cv2.fillPoly(a,[np.array([[40,40],[490,40],[490,190],[40,150]])],255)
        for x,y in ((105,185),(310,255),(445,215)):
            cv2.ellipse(a,(x,y-35),(30,35),0,0,360,255,-1)
        r,_=complete_contact_shadow(a,None)
        mirrored,_=complete_contact_shadow(a[:,::-1],None)
        np.testing.assert_allclose(r,mirrored[:,::-1],atol=1e-6)
        self.assertGreater(float(r[210,210]),.4)
        self.assertGreater(float(r[228,375]),.4)

    def test_bumper_overhang_keeps_shadow_at_tire_ground_height(self):
        import cv2
        a=np.zeros((340,600),np.uint8)
        a[40:180,60:530]=255
        for x in (150,410):
            cv2.ellipse(a,(x,190),(35,40),0,0,360,255,-1)
        r,_=complete_contact_shadow(a,None)
        # The bumper ends above the road; shadow must not follow it upward
        # and leave a bright hole underneath. Outside the bumper, the shadow
        # should fade quickly instead of forming an isolated dark lobe.
        self.assertGreater(float(r[230,515]),.20)
        self.assertLess(float(r[:,550:].max()),.01)
        self.assertLess(float(r[:,:40].max()),.01)
        self.assertLess(float(r[:130].max()),.001)
        self.assertLess(float(r[310:].max()),.01)

    def test_synthetic_shadow_fades_at_source_canvas_edges(self):
        import cv2
        a=np.zeros((260,500),np.uint8)
        a[20:170,8:490]=255
        for x in (70,410):
            cv2.ellipse(a,(x,195),(35,40),0,0,360,255,-1)
        r,layer=complete_contact_shadow(a,None)
        self.assertEqual(float(r[:,0].max()),0)
        self.assertEqual(float(r[:,-1].max()),0)
        self.assertEqual(float(r[-1].max()),0)
        self.assertLess(float(np.abs(r[:,1]-r[:,0]).max()),.01)
        self.assertLess(float(np.abs(r[-2]-r[-1]).max()),.01)
        self.assertGreater(float(r[220,250]),.4)

    def test_tires_have_dark_contact_without_a_gap(self):
        import cv2
        a=np.zeros((360,600),np.uint8)
        a[35:205,65:540]=255
        for x, y in ((145,240),(440,215)):
            cv2.ellipse(a,(x,y-40),(36,40),0,0,360,255,-1)
        r,_=complete_contact_shadow(a,None)
        for x, y in ((145,240),(440,215)):
            self.assertGreater(float(r[y+1,x]),.88)
            self.assertGreater(float(r[y+1,x]),float(r[y+12,x]))
        self.assertEqual(float(r[a>127].max()),0)

if __name__=='__main__':unittest.main()
