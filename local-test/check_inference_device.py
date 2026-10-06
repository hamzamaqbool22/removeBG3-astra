import os,sys,tempfile,types,unittest,runpy
from unittest.mock import patch,Mock
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vehicle_pipeline.segmentation import Segmenter
class Checks(unittest.TestCase):
 def session(self,device=None,available=True,fallback=False):
  ort=types.ModuleType('onnxruntime');ort.preload_dlls=Mock();ort.SessionOptions=lambda:types.SimpleNamespace();ort.ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0);ort.get_available_providers=lambda:['CUDAExecutionProvider'] if available else ['CPUExecutionProvider']
  class Session:
   @staticmethod
   def name():return 'birefnet-general'
   def __init__(self,model,opts,providers):self.inner_session=types.SimpleNamespace(get_providers=lambda:['CPUExecutionProvider'] if fallback else providers)
  sessions=types.ModuleType('rembg.sessions');sessions.sessions_class=[Session]
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'onnxruntime':ort,'rembg':types.ModuleType('rembg'),'rembg.sessions':sessions}):
   if device is not None:os.environ['INFERENCE_DEVICE']=device
   s=Segmenter(model_dir=d)
  return s,ort
 def test_cpu_default(self):
  s,ort=self.session();self.assertEqual(s.providers,['CPUExecutionProvider']);ort.preload_dlls.assert_not_called()
 def test_cuda_optin(self):
  s,ort=self.session('cuda');self.assertIn('CUDAExecutionProvider',s.providers);ort.preload_dlls.assert_called_once_with(directory='')
 def test_cuda_unavailable(self):
  with self.assertRaisesRegex(RuntimeError,'unavailable'):self.session('cuda',False)
 def test_no_silent_cpu_fallback(self):
  with self.assertRaisesRegex(RuntimeError,'refusing CPU fallback'):self.session('cuda',fallback=True)
 def test_invalid_device(self):
  with self.assertRaises(ValueError):self.session('automatic')
 def test_entrypoint_without_key(self):
  api=types.ModuleType('vehicle_pipeline.api');api.app=object()
  with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'vehicle_pipeline.api':api}):
   self.assertIs(runpy.run_path(str(ROOT/'main.py'))['app'],api.app)
 def test_existing_backend_auth_is_preserved(self):
  from vehicle_pipeline import api
  from fastapi import HTTPException
  request=types.SimpleNamespace(headers={})
  with patch.object(api,'API_KEY',''):
   api.authenticate(request)
  with patch.object(api,'API_KEY','test-secret'):
   with self.assertRaises(HTTPException):api.authenticate(request)
   api.authenticate(types.SimpleNamespace(headers={'authorization':'Bearer test-secret'}))
unittest.main()
