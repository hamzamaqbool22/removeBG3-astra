import {clip,quantile,median,round,leastSquares} from './numeric.js';
import {raster,resize,blur,op,median7,morph,components} from './cv.js';

export function recoverShadow(rgb,alpha){
  const scale=Math.min(1,1000/Math.max(rgb.w,rgb.h)),w=round(rgb.w*scale),h=round(rgb.h*scale),color=resize(rgb,w,h),a=resize(alpha,w,h),n=w*h,mask=Uint8Array.from(a.data,v=>v>127?1:0);
  let x0=w,x1=0,y0=h,y1=0;const bottom=new Int32Array(w).fill(-1);for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(mask[y*w+x]){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);bottom[x]=y;}
  if(x0>x1)return null;const bw=x1-x0+1,bh=y1-y0+1,floor=Float32Array.from(bottom,(_,x)=>Math.max(bottom[clip(x,x0,x1)],y0+.72*bh));
  const dist=op(raster(w,h,1,Uint8Array.from(mask,v=>1-v)),(a,b)=>cv.distanceTransform(a,b,cv.DIST_L2,5)).data,lum=new Float32Array(n),roi=new Uint8Array(n);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x;lum[i]=(color.data[i*3]*.2126+color.data[i*3+1]*.7152+color.data[i*3+2]*.0722)/255;roi[i]=+(x>x0-.22*bw&&x<x1+.22*bw&&y>=floor[x]-.18*bh&&y<floor[x]+.48*bh&&!mask[i]);}
  const smooth=blur(raster(w,h,1,lum),Math.max(.8,bw*.0015)).data,samples=[];
  for(let i=0;i<n;i++){const c=color.data.subarray(i*3,i*3+3);if(roi[i]&&dist[i]>.07*bh&&(Math.max(...c)-Math.min(...c))/255<.24&&lum[i]>.12&&lum[i]<.94)samples.push(i);}
  if(samples.length<300)return null;const step=Math.max(1,Math.floor(samples.length/16000)),rows=[],values=[];
  for(let s=0;s<samples.length;s+=step){const i=samples[s],x=(i%w-x0)/bw,y=(Math.floor(i/w)-y1)/bh;rows.push([1,x,y,x*y,x*x,y*y]);values.push(smooth[i]);}
  const q=quantile(values,.45);let valid=values.map(v=>v>=q),coeff,residual;
  for(let iteration=0;iteration<5;iteration++){coeff=leastSquares(rows,values,valid);if(!coeff)return null;residual=values.map((v,i)=>v-rows[i].reduce((s,x,j)=>s+x*coeff[j],0));const spread=Math.max(.018,median(residual.filter((_,i)=>valid[i]).map(Math.abs))*2.5);valid=residual.map(v=>v>-spread&&v<spread);if(valid.filter(Boolean).length<150)return null;}
  if(median(residual.filter((_,i)=>valid[i]).map(Math.abs))>.055)return null;
  let opacity=new Float32Array(n);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const xn=(x-x0)/bw,yn=(y-y1)/bh,base=clip(coeff[0]+coeff[1]*xn+coeff[2]*yn+coeff[3]*xn*yn+coeff[4]*xn*xn+coeff[5]*yn*yn,.12,.95);opacity[y*w+x]=clip(1-smooth[y*w+x]/base,0,.94);}
  let filtered=median7(raster(w,h,1,opacity));filtered=op(filtered,(a,b)=>cv.bilateralFilter(a,b,9,.12,4));filtered=blur(filtered,Math.max(.7,bw*.0012));
  let candidate=raster(w,h,1,Uint8Array.from(filtered.data,(v,i)=>{const x=i%w,y=Math.floor(i/w);return +(v>.16&&roi[i]&&x>=3&&x<w-3&&y>=3&&y<h-3);}));candidate=morph(morph(candidate,cv.MORPH_CLOSE,5,true),cv.MORPH_OPEN,5,true);
  const cc=components(candidate),attached=new Int32Array(cc.stats.length),acceptedIds=new Set();for(let i=0;i<n;i++)if(cc.labels[i]>0&&dist[i]<Math.max(5,.04*bh))attached[cc.labels[i]]++;
  for(let k=1;k<cc.stats.length;k++){const area=cc.stats[k][4];if(area>=Math.max(30,bw*bh*.001)&&area<=bw*bh*.5&&attached[k]>=Math.max(6,.007*bw))acceptedIds.add(k);}
  const accepted=Float32Array.from(cc.labels,v=>acceptedIds.has(v)?1:0);let area=0;for(const v of accepted)area+=v;if(area<bw*bh*.008)return null;
  opacity=morph(filtered,cv.MORPH_CLOSE,5,true).data;const support=blur(raster(w,h,1,accepted),Math.max(.65,bw*.001)).data,recovered=new Float32Array(n);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x,fade=clip(Math.min(x,y,w-1-x,h-1-y)/Math.max(3,.012*bw));recovered[i]=opacity[i]*support[i]*fade*fade*(3-2*fade);}
  const inside=morph(raster(w,h,1,recovered),cv.MORPH_DILATE).data;for(let i=0;i<n;i++)if(mask[i])recovered[i]=inside[i];return resize(raster(w,h,1,recovered),rgb.w,rgb.h,cv.INTER_LINEAR);
}
