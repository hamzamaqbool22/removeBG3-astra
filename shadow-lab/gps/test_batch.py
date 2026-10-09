import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('gps_batch', Path(__file__).with_name('batch.py'))
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def test_failure_does_not_stop_other_images_and_filenames_are_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            images = [Path(tmp)/'wall middle.png',Path(tmp)/'car.webp',Path(tmp)/'car.jpg']
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if '--image' in command and command[command.index('--image')+1] == str(images[1]):
                    raise subprocess.CalledProcessError(1,command)
            with patch.object(batch.subprocess,'run',side_effect=run):
                code = batch.run_batch(images,Path(tmp)/'background.png',folder)
            self.assertEqual(code,1)
            report=json.loads((folder/'summary.json').read_text())
            self.assertEqual([r['status'] for r in report['images']],['ok','failed','ok'])
            self.assertEqual(report['succeeded'],2)
            self.assertEqual(len(calls),5)
            self.assertIn(str(images[0]),calls[0])
            with (folder/'timings.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),3)
            self.assertIsNotNone(report['finished_at'])
            self.assertEqual(len({r['result_folder'] for r in report['images']}),3)
            for row in report['images']:
                self.assertGreaterEqual(row['total_seconds'],0)
                self.assertTrue(row['started_at'] and row['finished_at'])

    def test_interruption_writes_summary_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with patch.object(batch.subprocess,'run',side_effect=KeyboardInterrupt):
                code=batch.run_batch([Path('first.jpg'),Path('second.jpg')],Path('bg.png'),folder)
            self.assertEqual(code,130)
            report=json.loads((folder/'summary.json').read_text())
            self.assertEqual(len(report['images']),1)
            self.assertEqual(report['images'][0]['status'],'interrupted')
            self.assertTrue(report['finished_at'])


if __name__=='__main__':
    unittest.main()
