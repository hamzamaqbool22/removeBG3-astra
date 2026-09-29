import {explainFailure} from './src/errors.js';
import {clearSavedAssets} from './src/assets.js';
const $=id=>document.getElementById(id);
let worker=null,active=false,urls=[];
function log(message,details){const line=`${new Date().toLocaleTimeString()} ${message}${details?' '+JSON.stringify(details):''}`;console.info('[Vehicle Local]',message,details??'');$('log').textContent=($('log').textContent+line+'\n').split('\n').slice(-250).join('\n');$('log').scrollTop=$('log').scrollHeight;}
$('copyLog').onclick=async()=>{try{await navigator.clipboard.writeText($('log').textContent);$('copyLog').textContent='Copied';}catch{log('Select and copy the log text manually.');}};
log('Panel ready — v0.4.0, BiRefNet Lite 512. Photos are processed locally.');
if(location.protocol==='file:'){$('status').textContent='Load this folder in chrome://extensions → Developer mode → Load unpacked, then click the extension icon.';$('generate').disabled=true;log('Direct file opening is unsupported; load as a Chrome extension.');}
const catalog=await fetch('backgrounds/catalog.json').then(r=>r.json()).catch(()=>({}));
$('folder').addEventListener('change',()=>{const f=$('folder').value;$('numberLabel').hidden=!catalog[f];$('customLabel').hidden=f!=='custom';$('number').replaceChildren(...(catalog[f]??[]).map(n=>new Option(n,n)));});
function stop(){worker?.terminate();worker=null;active=false;$('generate').disabled=false;$('cancel').disabled=true;$('progress').hidden=true;}
$('clearCache').onclick=async()=>{stop();await clearSavedAssets();$('status').textContent='Saved downloads cleared. The next run will download its model again.';log('Saved models and backgrounds cleared. Bundled defaults are unchanged.');};
$('cancel').onclick=()=>{stop();$('status').textContent='Stopped. Browser model memory released.';log('Stopped; worker terminated.');};
$('generate').onclick=async()=>{
  if(active)return;const files=[...$('photos').files];if(!files.length){$('status').textContent='Choose at least one vehicle photo.';return;}
  $('notice').textContent='';
  const folder=$('folder').value;let background=null,backgroundKey=null;
  try{if(catalog[folder]){const number=$('number').value;if(!number)throw Error('Select a background number.');backgroundKey=`${folder}/${number}`;}else if(folder==='custom'){background=$('custom').files[0];if(!background)throw Error('Choose a background image.');}}
  catch(e){$('status').textContent=e.message;return;}
  urls.forEach(URL.revokeObjectURL);urls=[];$('results').replaceChildren();$('details').textContent='';active=true;$('generate').disabled=true;$('cancel').disabled=false;$('progress').hidden=false;
  $('status').textContent='Starting local processing…';
  log('Batch started',{photos:files.length,background:folder,enhancement:$('enhance').checked,device:$('backend').value});
  if(!worker){worker=new Worker('worker.js');worker.onerror=e=>{stop();$('status').textContent=`Browser worker failed: ${e.message}. Try reopening the panel.`;};}
  worker.onmessage=({data})=>{
    if(data.type==='result')log('Photo completed',{name:data.name,seconds:Number(data.seconds.toFixed(2)),shadow:data.shadowMode});
    if(data.type==='log')log(data.message,data.details);
    else if(data.type==='warning')$('notice').textContent=data.message;
    else if(data.type==='progress')$('status').textContent=data.message;
    else if(data.type==='result'){const url=URL.createObjectURL(data.blob);urls.push(url);const article=document.createElement('article'),title=document.createElement('strong'),im=document.createElement('img'),link=document.createElement('a'),caption=document.createElement('p');title.textContent=data.name;im.src=url;im.alt=`Processed ${data.name}`;link.href=url;link.download=data.name.replace(/\.[^.]+$/,'')+'-local.png';link.textContent='Download PNG';caption.textContent=`${data.seconds.toFixed(1)} seconds · ${data.shadowMode}`;article.append(title,im,caption,link);$('results').append(article);$('details').textContent+=JSON.stringify(data.info,null,2)+'\n';}
    else if(data.type==='error'){$('status').textContent=`Local processing failed: ${explainFailure(data.detail??data.message)} No photo was sent to a server.`;log('Technical error',{detail:data.detail??data.message});stop();}
    else if(data.type==='done'){active=false;$('generate').disabled=false;$('progress').hidden=true;$('status').textContent='Finished. Model stays loaded for the next photo. Stop to release memory.';log('Batch finished successfully.');}
  };
  worker.postMessage({type:'generate',files,background,backgroundKey,white:folder==='white',enhancement:$('enhance').checked,backend:$('backend').value});
};
