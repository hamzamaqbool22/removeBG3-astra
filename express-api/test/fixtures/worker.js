import {writeFile} from 'node:fs/promises';
process.send({type:'ready'});
process.on('message',async job=>{
  if(job.type!=='job')return;
  if(job.options.crash){process.exit(1);return;}
  if(job.options.hang)return;
  await new Promise(r=>setTimeout(r,job.options.delay??5));
  await writeFile(job.output,'test');process.send({type:'done',id:job.id,result:{}});
});
process.on('disconnect',()=>process.exit());
