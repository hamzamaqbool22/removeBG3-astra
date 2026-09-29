import {createRequire} from 'node:module';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';
import {config} from './config.js';
import {modelPath,backgroundPath} from './assets.js';
import {raster,resize,largest,crop} from './pipeline/cv.js';
import {lanczos} from './pipeline/numeric.js';
import {processWithMask} from './pipeline/pipeline.js';
sharp.cache(false);sharp.concurrency(1);
let session;
export async function initialize(){
  globalThis.cv=await createRequire(import.meta.url)('@techstark/opencv-js');
  session=await ort.InferenceSession.create(await modelPath(),{executionProviders:['cpu'],intraOpNumThreads:config.threads,interOpNumThreads:1,graphOptimizationLevel:'all',enableCpuMemArena:false,enableMemPattern:false});
}
async function decode(file){
  const {data,info}=await sharp(file instanceof URL?file.pathname:file,{limitInputPixels:config.maxPixels,animated:false}).rotate().toColourspace('srgb').removeAlpha().raw().toBuffer({resolveWithObject:true});
  return raster(info.width,info.height,3,new Uint8Array(data));
}
async function predict(rgb){
  const small=lanczos(rgb,512,512,'bilinear'),plane=512*512,input=new Float32Array(plane*3);
  for(let i=0;i<plane;i++)for(let c=0;c<3;c++)input[c*plane+i]=(small.data[i*3+c]/255-[.485,.456,.406][c])/[.229,.224,.225][c];
  const tensor=new ort.Tensor('float32',input,[1,3,512,512]);let outputs;
  try{
    outputs=await session.run({[session.inputNames[0]]:tensor});const values=outputs[session.outputNames[0]].data,mask=new Float32Array(plane);let max=0;
    for(let i=0;i<plane;i++){if(!Number.isFinite(values[i]))throw Error('Invalid model output');mask[i]=1/(1+Math.exp(-values[i]));max=Math.max(max,mask[i]);}
    if(max<.5)throw Error('No foreground detected');
    const resized=resize(raster(512,512,1,mask),rgb.w,rgb.h,cv.INTER_LINEAR);
    return raster(rgb.w,rgb.h,1,Uint8Array.from(resized.data,v=>Math.max(0,Math.min(255,Math.round(v*255)))));
  }finally{tensor.dispose();if(outputs)for(const t of Object.values(outputs))t.dispose();}
}
export async function processImage(input,output,options,progress=()=>{}){
  progress('segmentation');const rgb=await decode(input);let mask=await predict(rgb);
  const comp=largest(mask),[x,y,x1,y1]=comp.bbox,w=x1-x,h=y1-y;
  if(comp.area>rgb.w*rgb.h*.01&&Math.max(w/rgb.w,h/rgb.h)<.60){
    const pad=Math.max(16,Math.round(.2*Math.max(w,h))),xx=Math.max(0,x-pad),yy=Math.max(0,y-pad),right=Math.min(rgb.w,x1+pad),bottom=Math.min(rgb.h,y1+pad);
    const detail=await predict(crop(rgb,xx,yy,right-xx,bottom-yy));mask=raster(rgb.w,rgb.h,1,new Uint8Array(rgb.w*rgb.h));
    for(let j=0;j<detail.h;j++)mask.data.set(detail.data.subarray(j*detail.w,(j+1)*detail.w),(j+yy)*rgb.w+xx);
  }
  progress('background');const background=await backgroundPath(options);
  const result=processWithMask(rgb,mask,background.file?await decode(background.file):null,options.enhancement,progress);
  const image=result.image;
  await sharp(Buffer.from(image.data),{raw:{width:image.w,height:image.h,channels:image.c}}).png().toFile(output);
  return {warning:background.warning??null,shadowMode:result.shadowMode};
}
