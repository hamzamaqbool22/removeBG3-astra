"""CPU-only regression tests; no real checkpoints or GPU needed."""
import io
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from checkpoint import load_checkpoint


class UnexpectedMetadata:
    pass


class CheckpointTests(unittest.TestCase):
    def test_current_and_legacy_numpy_metadata(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'regressor.pth'
                original = {'net': {'weight': torch.tensor([1., 2.])},
                            'best_loss': np.float64(.125), 'epoch': np.int64(3),
                            'score32': np.float32(.5)}
                buffer = io.BytesIO()
                torch.save(original, buffer)
                with zipfile.ZipFile(buffer) as src, zipfile.ZipFile(path, 'w') as dst:
                    for item in src.infolist():
                        data = src.read(item.filename)
                        if legacy and item.filename.endswith('data.pkl'):
                            self.assertIn(b'numpy._core.multiarray', data)
                            data = data.replace(b'numpy._core.multiarray', b'numpy.core.multiarray')
                        dst.writestr(item, data)
                globals_before = torch.serialization.get_safe_globals().copy()
                result = load_checkpoint(path)
                torch.testing.assert_close(result['net']['weight'], original['net']['weight'])
                self.assertEqual(result['best_loss'], .125)
                self.assertEqual(result['epoch'], 3)
                self.assertEqual(result['score32'], .5)
                self.assertEqual(torch.serialization.get_safe_globals(), globals_before)

    def test_unknown_metadata_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unexpected.pth'
            torch.save({'net': {}, 'metadata': UnexpectedMetadata()}, path)
            before = torch.serialization.get_safe_globals().copy()
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(path)
            self.assertEqual(torch.serialization.get_safe_globals(), before)


if __name__ == '__main__':
    unittest.main()
