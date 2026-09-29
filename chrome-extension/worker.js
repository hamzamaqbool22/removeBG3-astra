// Dedicated worker: all inference and image processing happen locally, off the UI thread.
let initialized=null,session=null,provider=null,busy=false,gpuDevice=null;
const log=(message,details)=>{console.info('[Vehicle Local]',message,details??'');postMessage({type:'log',message,details});};
const progress=message=>{log(message);postMessage({type:'progress',message});};
async function libraries(){
  if(!initialized)initialized=(async()=>{importScripts('vendor/opencv.js','vendor/ort.webgpu.min.js');globalThis.cv=await globalThis.cv;ort.env.wasm.wasmPaths=new URL('vendor/',self.location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;return Promise.all([import('./src/cv.js'),import('./src/numeric.js'),import('./src/geometry.js'),import('./src/pipeline.js')]);})();
  return initialized;
}
async function decode(blob){const bitmap=await createImageBitmap(blob,{imageOrientation:'from-image',premultiplyAlpha:'none',colorSpaceConversion:'none'});try{if(bitmap.width*bitmap.height>20_000_000)throw Error('Photo exceeds 20 megapixels.');const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0);const rgba=ctx.getImageData(0,0,canvas.width,canvas.height),data=new Uint8Array(canvas.width*canvas.height*3);for(let i=0;i<data.length/3;i++)for(let c=0;c<3;c++)data[i*3+c]=rgba.data[i*4+c];return {w:canvas.width,h:canvas.height,c:3,data};}finally{bitmap.close();}}
async function loadModel(backend){
  if(session&&provider===backend)return;
  if(session){await session.release();session=null;}
  gpuDevice?.destroy();gpuDevice=null;
  let device;
  if(backend==='webgpu'){
    const adapter=await navigator.gpu?.requestAdapter();
    if(!adapter)throw Error('WebGPU is unavailable on this browser/device. Update Chrome and enable hardware acceleration.');
    if(!adapter.features.has('shader-f16'))throw Error('This GPU does not support the model’s FP16 calculations. Choose Local CPU.');
    const requiredLimits={};
    for(const key of ['maxStorageBuffersPerShaderStage','maxStorageBufferBindingSize','maxBufferSize','maxComputeWorkgroupStorageSize','maxComputeInvocationsPerWorkgroup','maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupSizeZ','maxComputeWorkgroupsPerDimension'])requiredLimits[key]=adapter.limits[key];
    device=await adapter.requestDevice({requiredLimits,requiredFeatures:['shader-f16','subgroups'].filter(f=>adapter.features.has(f))});
    gpuDevice=device;
    log('GPU ready',{maxBufferMB:Math.round(adapter.limits.maxBufferSize/1048576),storageBuffers:adapter.limits.maxStorageBuffersPerShaderStage,fp16:true});
  }
  const model=backend==='webgpu'?'birefnet-lite-512-fp16':'birefnet-lite-512';
  progress(`Loading BiRefNet Lite locally (${backend==='webgpu'?'FP16, 94 MB':'FP32, 183 MB'})…`);
  log('Model configuration',{backend,model,precision:backend==='webgpu'?'float16':'float32',resolution:512});
  const started=performance.now();
  const {modelBytes}=await import('./src/assets.js');
  const bytes=await modelBytes(`${model}.onnx`,progress);
  progress('Preparing the downloaded model for local processing…');
  session=await ort.InferenceSession.create(bytes,{executionProviders:[device?{name:'webgpu',device}:backend],graphOptimizationLevel:'all',enableCpuMemArena:false,enableMemPattern:false});provider=backend;
  log('Model ready',{seconds:((performance.now()-started)/1000).toFixed(2)});
}
async function predict(rgb,modules){
  const [{raster,resize},{lanczos}]=modules,small=lanczos(rgb,512,512,'bilinear'),plane=512*512,means=[.485,.456,.406],std=[.229,.224,.225],input=new Float32Array(plane*3);
  for(let i=0;i<plane;i++)for(let c=0;c<3;c++)input[c*plane+i]=(small.data[i*3+c]/255-means[c])/std[c];
  log('Inference started',{width:rgb.w,height:rgb.h,modelInput:'512 × 512'});
  const tensor=new ort.Tensor('float32',input,[1,3,512,512]);let outputs;
  try{outputs=await session.run({[session.inputNames[0]]:tensor});const logits=outputs[session.outputNames[0]],values=await logits.getData(),mask=new Float32Array(plane);let max=0;for(let i=0;i<plane;i++){if(!Number.isFinite(values[i]))throw Error('Model returned invalid pixel values.');mask[i]=1/(1+Math.exp(-values[i]));max=Math.max(max,mask[i]);}if(max<.5)throw Error('No foreground detected.');const resized=resize(raster(512,512,1,mask),rgb.w,rgb.h,cv.INTER_LINEAR);return raster(rgb.w,rgb.h,1,Uint8Array.from(resized.data,v=>Math.max(0,Math.min(255,Math.round(v*255)))));}
  finally{tensor.dispose();if(outputs)for(const t of Object.values(outputs))t.dispose();}
}
async function segment(rgb,modules){const [{largest,crop,raster}]=modules;let mask=await predict(rgb,modules),passes=1;const comp=largest(mask),[x,y,x1,y1]=comp.bbox,w=x1-x,h=y1-y;
  if(comp.area>rgb.w*rgb.h*.01&&Math.max(w/rgb.w,h/rgb.h)<.60){progress('Refining distant-vehicle details');const pad=Math.max(16,Math.round(.2*Math.max(w,h))),xx=Math.max(0,x-pad),yy=Math.max(0,y-pad),right=Math.min(rgb.w,x1+pad),bottom=Math.min(rgb.h,y1+pad),detail=await predict(crop(rgb,xx,yy,right-xx,bottom-yy),modules);mask=raster(rgb.w,rgb.h,1,new Uint8Array(rgb.w*rgb.h));for(let j=0;j<detail.h;j++)mask.data.set(detail.data.subarray(j*detail.w,(j+1)*detail.w),(j+yy)*rgb.w+xx);passes=2;}
  return {mask,passes};
}
async function png(image,white){const canvas=new OffscreenCanvas(image.w,image.h),ctx=canvas.getContext('2d'),rgba=new Uint8ClampedArray(image.w*image.h*4);for(let i=0;i<image.w*image.h;i++){const a=image.c===4?image.data[i*4+3]/255:1;for(let c=0;c<3;c++){const v=image.data[i*image.c+c];rgba[i*4+c]=white?v*a+255*(1-a):v;}rgba[i*4+3]=white?255:Math.round(a*255);}ctx.putImageData(new ImageData(rgba,image.w,image.h),0,0);return canvas.convertToBlob({type:'image/png'});}
self.onmessage=async({data})=>{
  if(data.type!=='generate'||busy)return;
  busy=true;
  let stage='initialization';
  const started=performance.now();
  const heartbeat=setInterval(()=>log('Still processing',{stage,elapsedSeconds:Math.round((performance.now()-started)/1000)}),15000);
  try{
    const modules=await libraries();
    log('Runtime loaded',{onnx:ort.env.versions,backend:data.backend});
    stage='loading model';await loadModel(data.backend);
    const {backgroundBlob}=await import('./src/assets.js');
    const blob=data.backgroundKey?await backgroundBlob(data.backgroundKey,message=>{progress(message);if(message.startsWith('Using fallback'))postMessage({type:'warning',message});}):data.background;
    const background=blob?await decode(blob):null;
    for(const file of data.files){
      const photoStarted=performance.now();
      stage='background removal';progress(`Removing background: ${file.name}`);
      const rgb=await decode(file),{mask,passes}=await segment(rgb,modules);
      if(data.diagnostics)postMessage({type:'diagnostic-mask',name:file.name,width:mask.w,height:mask.h,mask:mask.data});
      log('Segmentation completed',{passes,seconds:((performance.now()-photoStarted)/1000).toFixed(2)});
      stage='shadows and compositing';
      const result=modules[3].processWithMask(rgb,mask,background,data.enhancement,progress);
      stage='encoding PNG';
      const blob=await png(result.image,data.white);
      postMessage({type:'result',name:file.name,blob,seconds:(performance.now()-photoStarted)/1000,shadowMode:result.shadowMode,info:{backend:provider,inferencePasses:passes,geometry:result.geometry}});
    }
    postMessage({type:'done'});
  }catch(e){
    const detail=String(e?.message??e);
    console.error('[Vehicle Local]',stage,e);
    log('Processing failed',{stage,detail,elapsedSeconds:((performance.now()-started)/1000).toFixed(2)});
    postMessage({type:'error',detail});
    if(session){try{await session.release();}catch{}session=null;}
    gpuDevice?.destroy();gpuDevice=null;
  }finally{clearInterval(heartbeat);busy=false;}
};
