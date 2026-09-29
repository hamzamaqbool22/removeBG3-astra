import {readFile,readdir,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
const args=process.argv.slice(2),get=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const dir=get('--input'),count=Number(get('--count','100')),base=get('--url','http://127.0.0.1:8002'),out=get('--output','validation/load');
if(!dir||!Number.isInteger(count)||count<1||count>1000)throw Error('Usage: npm run load -- --input /folder --count 100 [--url http://127.0.0.1:8002]');
const files=(await readdir(dir)).filter(f=>/\.(png|jpe?g|webp)$/i.test(f));if(!files.length)throw Error('No photos found');
await mkdir(out,{recursive:true});
const headers=process.env.API_KEY?{Authorization:`Bearer ${process.env.API_KEY}`}:{},start=Date.now();
const submitted=await Promise.all(Array.from({length:count},async(_,i)=>{
  const name=files[i%files.length],body=new FormData();body.append('image',new Blob([await readFile(path.join(dir,name))]),name);body.append('isBackgroundWant','false');
  const response=await fetch(base+'/jobs',{method:'POST',headers,body,signal:AbortSignal.timeout(120000)});const result=await response.json();
  return {index:i,name,http:response.status,...result};
}));
console.log(`Accepted ${submitted.filter(j=>j.http===202).length}/${count}; all submitted concurrently.`);
const pending=new Map(submitted.filter(j=>j.http===202).map(j=>[j.jobId,j]));
const deadline=Date.now()+3600000;
while(pending.size){
  if(Date.now()>deadline)throw Error('Load test timed out');
  await new Promise(r=>setTimeout(r,2000));
  await Promise.all([...pending].map(async([id,job])=>{
    const r=await fetch(base+job.statusUrl,{headers});if(!r.ok)throw Error(`Poll failed ${r.status}`);const status=await r.json();
    if(['done','failed'].includes(status.status)){
      Object.assign(job,status,{elapsedSeconds:(Date.now()-start)/1000});pending.delete(id);
      if(status.status==='done'){const image=await fetch(base+job.resultUrl,{headers});if(!image.ok)throw Error('Result download failed');await writeFile(path.join(out,`${job.index}.png`),Buffer.from(await image.arrayBuffer()));}
      console.log(`${job.index+1}: ${job.name} ${status.status} after ${job.elapsedSeconds.toFixed(1)}s`);
    }
  }));
}
await writeFile(path.join(out,'report.json'),JSON.stringify(submitted,null,2));
console.log(`Finished ${submitted.filter(j=>j.status==='done').length}/${count} in ${((Date.now()-start)/1000).toFixed(1)}s`);
if(submitted.some(j=>j.status!=='done'))process.exitCode=1;
