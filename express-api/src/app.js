import express from 'express';
import multer from 'multer';
import {randomUUID,timingSafeEqual} from 'node:crypto';
import {rename,rm,statfs} from 'node:fs/promises';
import path from 'node:path';
import {config} from './config.js';
import {fileURLToPath} from 'node:url';
import {catalog} from './assets.js';
import {downloadImage} from './download.js';
const error=(status,message)=>Object.assign(Error(message),{status});
function boolean(value,fallback){if(value===undefined)return fallback;if(value===true||value==='true')return true;if(value===false||value==='false')return false;throw error(400,'Boolean values must be true or false');}
export function options(body={}){
  const isBackgroundWant=boolean(body.isBackgroundWant,false),enhancement=boolean(body.enhancement??body.enhancment,false);
  const folder=body.folder??body.backgroundFolder??'parking-lots';
  const background=String(body.background??'1').replace(/\.png$/,'')+'.png';
  if(!['parking-lots','backgrounds'].includes(folder)||!catalog.backgrounds[`${folder}/${background}`])throw error(400,'Unknown background folder or number');
  return {isBackgroundWant,enhancement,folder,background};
}
export function createApp(queue){
  const app=express();app.disable('x-powered-by');
  const origins=(process.env.CORS_ORIGINS??'').split(',').filter(Boolean);
  app.use((req,res,next)=>{
    const origin=req.headers.origin;
    if(origin&&origins.includes(origin)){res.set('Access-Control-Allow-Origin',origin);res.vary('Origin');res.set('Access-Control-Allow-Methods','GET,POST,OPTIONS');res.set('Access-Control-Allow-Headers','Authorization,Content-Type');}
    if(req.method==='OPTIONS')return res.sendStatus(origin&&!origins.includes(origin)?403:204);
    res.set('Cache-Control','no-store');next();
  });
  app.get('/health',(_req,res)=>res.json({status:'ok',workerReady:queue.ready}));
  app.get('/ready',(_req,res)=>res.status(queue.ready?200:503).json({ready:queue.ready}));
  app.use((req,res,next)=>{
    const secret=process.env.API_KEY;
    if(secret){const expected=Buffer.from(`Bearer ${secret}`),given=Buffer.from(req.headers.authorization??'');if(expected.length!==given.length||!timingSafeEqual(expected,given))return res.status(401).json({error:'Unauthorized'});}
    next();
  });
  app.get('/backgrounds',(_req,res)=>res.json({folders:Object.fromEntries(['parking-lots','backgrounds'].map(folder=>[folder,Object.keys(catalog.backgrounds).filter(k=>k.startsWith(folder+'/')).map(k=>k.split('/')[1])]))}));
  app.get('/openapi.json',(_req,res)=>res.sendFile(fileURLToPath(new URL('../assets/openapi.json',import.meta.url))));
  const multipart=multer({dest:path.join(queue.dir,'tmp'),limits:{fileSize:config.maxBytes,files:1,fields:8,fieldSize:2048,parts:9}}).single('image');
  const json=express.json({limit:'8kb'});
  const parse=req=>req.is('multipart/form-data')?multipart:req.is('application/json')?json:null;
  app.post(['/jobs','/process'],async(req,res,next)=>{
    // Reservation is synchronous and precedes parsing, so simultaneous arrivals
    // cannot all observe the same free slot. Disk uploads do not hold image buffers.
    if(queue.db.prepare('SELECT count(*) n FROM jobs').get().n+queue.reservations>=1000||!queue.reserve())return res.set('Retry-After','10').status(429).json({error:'Queue or retained-job storage is full. Retry later.'});
    const id=randomUUID(),input=path.join(queue.dir,'inputs',id);let accepted=false;
    try{
      const disk=await statfs(queue.dir);
      if(Number(disk.bavail)*Number(disk.bsize)<1024**3+(queue.reservations*config.maxBytes))throw error(503,'Server storage is low. Retry later.');
      const parser=parse(req);if(!parser)throw error(415,'Use multipart/form-data or application/json');
      await new Promise((resolve,reject)=>parser(req,res,e=>e?reject(e):resolve()));
      const opts=options(req.body);
      const url=req.body?.imageurl;
      if(Boolean(req.file)===Boolean(url))throw error(400,'Provide exactly one image upload or imageurl');
      if(req.file)await rename(req.file.path,input);
      else{
        if(typeof url!=='string'||url.length>2048)throw error(400,'Invalid imageurl');
        try{await downloadImage(url,input,config.maxBytes,(process.env.IMAGE_URL_HOSTS??'').split(',').filter(Boolean));}catch(e){throw error(400,e.message);}
      }
      if(req.aborted)throw error(400,'Upload disconnected');
      queue.add(id,opts);accepted=true;
      res.status(202).location(`/jobs/${id}`).json({jobId:id,status:'queued',statusUrl:`/jobs/${id}`,resultUrl:`/jobs/${id}/result`,pollAfterMs:2000});
    }catch(e){next(e);}finally{
      queue.release();
      if(!accepted)await rm(input,{force:true});
      if(req.file)await rm(req.file.path,{force:true});
    }
  });
  const job=(req,res,next)=>{
    if(!/^[a-f0-9-]{36}$/.test(req.params.id))return res.status(404).json({error:'Job not found'});
    const info=queue.get(req.params.id);if(!info)return res.status(404).json({error:'Job not found or expired'});req.job=info;next();
  };
  app.get('/jobs/:id',job,(req,res)=>res.json({...req.job,resultUrl:req.job.status==='done'?`/jobs/${req.job.id}/result`:null,pollAfterMs:2000}));
  app.get('/jobs/:id/result',job,(req,res,next)=>{
    if(req.job.status!=='done')return res.status(409).json({error:'Result is not ready',status:req.job.status});
    res.type('png').set('Content-Disposition',`attachment; filename="${req.job.id}.png"`).sendFile(path.join(queue.dir,'results',req.job.id+'.png'),e=>{if(e)next(e);});
  });
  app.use((err,_req,res,_next)=>{
    if(res.headersSent)return res.destroy();
    const status=err instanceof multer.MulterError?(err.code==='LIMIT_FILE_SIZE'?413:400):err.status??500;
    if(status===500)console.error(err);
    res.status(status).json({error:status===500?'Internal server error':err.message});
  });
  return app;
}
