"""Regression checks on the supplied set plus scale/translation invariants.

The cached masks make these tests quick and independent of network/model load.
They validate geometry/rendering, not segmentation on unseen photographs.
"""
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image

from vehicle_pipeline.pipeline import VehiclePipeline
from vehicle_pipeline.segmentation import refine_mask

ROOT = Path(__file__).resolve().parents[1]


class PipelineRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        pipe = VehiclePipeline()
        for p in sorted((ROOT / 'images').glob('*.png')):
            mask_path = ROOT / 'outputs/segmentation/birefnet-general' / p.stem / 'mask.png'
            if not mask_path.exists():
                raise unittest.SkipTest('Run the BiRefNet benchmark to produce cached test masks')
            with Image.open(p) as image:
                cls.results[p.stem] = pipe.process(image, np.array(Image.open(mask_path)))

    def test_all_views_have_grounded_contacts_and_preserved_interior(self):
        self.assertEqual(len(self.results), 8)
        for name, result in self.results.items():
            with self.subTest(view=name):
                g = result.metadata['normalized_geometry']
                self.assertEqual(len(g['contacts']), 2)
                self.assertAlmostEqual(max(c['y'] for c in g['contacts']), .80 * 768, places=4)
                for contact in g['contacts']:
                    self.assertTrue(0 <= contact['confidence'] <= 1)
                    x, y = round(contact['x']), round(contact['y'])
                    shadow = result.stages['shadow_tire_contact_alpha']
                    self.assertGreater(int(shadow[max(0,y-2):y+3,max(0,x-2):x+3].max()), 180)
                a = result.stages['normalized_alpha']
                opaque = cv2.erode((a == 255).astype(np.uint8), np.ones((5,5),np.uint8)).astype(bool)
                color = result.stages['normalized_cutout'][...,:3]
                self.assertEqual(int(np.abs(result.final.astype(int)-color.astype(int))[opaque].max()), 0)
                self.assertTrue(np.all(result.final[:10] == 255))
                self.assertTrue(np.all(result.final[-10:] == 255))
                self.assertTrue(np.all(result.final[:,:10] == 255))
                self.assertTrue(np.all(result.final[:,-10:] == 255))

    def test_hidden_wheels_do_not_become_grille_detections(self):
        for name in ('front', 'back'):
            g = self.results[name].metadata['geometry']
            self.assertEqual(g['view_cues']['support_mode'], 'occluded_axle_pair')
            self.assertTrue(all('ellipse' not in c for c in g['contacts']))
        for name in set(self.results) - {'front','back'}:
            g = self.results[name].metadata['geometry']
            self.assertEqual(g['view_cues']['support_mode'], 'visible_wheel_pair')

    def test_framing_preserves_proportions_without_filling_head_on_width(self):
        widths = {}
        for name, result in self.results.items():
            raw = result.metadata['geometry']['bbox']
            placed = result.metadata['normalized_geometry']['bbox']
            sw, sh = raw[2]-raw[0], raw[3]-raw[1]
            dw, dh = placed[2]-placed[0], placed[3]-placed[1]
            self.assertAlmostEqual(sw/sh, dw/dh, places=5)
            self.assertLessEqual(dw, .88*1024 + 1e-3)
            self.assertLessEqual(dh, .46*768 + 1e-3)
            widths[name] = dw
        self.assertLess(widths['front'], widths['left'] * .75)

    def test_zoomed_out_input_is_normalized_to_same_visible_size(self):
        # Exercise the whole geometry/rendering stack after rescaling a source
        # into a larger scene. Cached alpha isolates framing from model error.
        pipe = VehiclePipeline()
        for name in ('front','frontLeft','left','rearRight'):
            with self.subTest(view=name):
                rgb=np.array(Image.open(ROOT/'images'/f'{name}.png').convert('RGB'))
                mask=np.array(Image.open(ROOT/'outputs/segmentation/birefnet-general'/name/'mask.png'))
                matrix=np.array([[.72,0,113],[0,.72,67]],np.float32)
                size=(rgb.shape[1],rgb.shape[0])
                small=cv2.warpAffine(rgb,matrix,size,borderValue=(180,180,180))
                a=cv2.warpAffine(mask,matrix,size)
                result=pipe.process(Image.fromarray(small),a)
                original=self.results[name]
                b=result.metadata['normalized_geometry']['bbox']
                expected=original.metadata['normalized_geometry']['bbox']
                self.assertLess(max(abs(np.array(b)-expected)), 7.)
                self.assertAlmostEqual(result.metadata['normalized_geometry']['ground_y'],614.4,places=4)

    def test_empty_foreground_fails_explicitly(self):
        with self.assertRaises(ValueError):
            refine_mask(np.zeros((128,128),np.uint8))


if __name__ == '__main__':
    unittest.main()
