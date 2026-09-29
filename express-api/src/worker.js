import {initialize,processImage} from './processor.js';
await initialize();process.send({type:'ready'});
let busy=false;
process.on('message',async job=>{
  if(job.type!=='job'||busy)return;busy=true;
  try{const result=await processImage(job.input,job.output,job.options,stage=>process.send({type:'progress',id:job.id,stage}));process.send({type:'done',id:job.id,result});}
  catch(e){process.send({type:'failed',id:job.id,error:e.message});}
  finally{busy=false;}
});
// Never leave an orphan inference worker after its API parent dies.
process.on('disconnect',()=>process.exit(0));
