import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import gpu_batch as batch

class GpuBatchTests(unittest.TestCase):
    def fixture(self, folder):
        rows=[]
        for index,name in enumerate(('wall middle.png','bad.jpg','car.webp')):
            row=dict.fromkeys(batch.FIELDS,'')
            row.update(image=name,status='pending',prepare_seconds=0.,inference_seconds=0.,total_seconds=0.,result_folder=str(folder/str(index)/'result'))
            rows.append(row)
        manifest=dict(rows=rows,started_at=batch.now(),background='bg.png',samples=1,steps=50,seed=42)
        path=folder/'manifest.json'
        path.write_text(json.dumps(manifest))
        return path,manifest

    def test_reuses_sessions_and_continues_after_bad_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            path,manifest=self.fixture(folder)
            session=SimpleNamespace(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            factory=Mock(return_value=session)
            prepare=Mock()
            rows,startup=batch.prepare_all(path,factory,prepare)
            factory.assert_called_once()
            self.assertEqual(prepare.call_count,3)
            self.assertTrue(all(call.args[3] is session for call in prepare.call_args_list))
            runner=Mock()
            runner.run.side_effect=[None,ValueError('empty vehicle mask'),None]
            runner_factory=Mock(return_value=runner)
            batch.infer_all(folder,rows,manifest,startup,runner_factory)
            runner_factory.assert_called_once()
            self.assertEqual(runner.run.call_count,3)
            self.assertTrue(all(call.args[2:]==(1,50,42) for call in runner.run.call_args_list))
            report=json.loads((folder/'summary.json').read_text())
            self.assertEqual([r['status'] for r in report['images']],['ok','failed','ok'])
            self.assertEqual(set(report['model_startup_seconds']),{'segmentation','gps'})
            for row in report['images']:
                self.assertAlmostEqual(row['total_seconds'],row['prepare_seconds']+row['inference_seconds'],places=3)
                self.assertTrue(row['finished_at'])

    def test_cpu_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.fixture(Path(tmp))
            prepare=Mock()
            with self.assertRaisesRegex(RuntimeError,'CPU fallback'):
                batch.prepare_all(path,lambda:SimpleNamespace(providers=['CPUExecutionProvider']),prepare)
            prepare.assert_not_called()

    def test_cuda_failure_stops_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            path,manifest=self.fixture(folder)
            rows,startup=batch.prepare_all(path,lambda:SimpleNamespace(providers=['CUDAExecutionProvider']),Mock())
            runner=Mock()
            runner.run.side_effect=RuntimeError('CUDA out of memory')
            with self.assertRaisesRegex(RuntimeError,'GPU failure'):
                batch.infer_all(folder,rows,manifest,startup,lambda:runner)
            self.assertEqual(runner.run.call_count,1)
            self.assertEqual(rows[0]['status'],'failed')
            self.assertEqual(rows[1]['status'],'prepared')

if __name__=='__main__':
    unittest.main()
