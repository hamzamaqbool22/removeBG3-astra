import {clip,mean,median,interp,u8} from './numeric.js';
import {raster,warp,resize} from './cv.js';

export function normalize(rgb,alpha,geometry){
  const [x0,y0,x1,y1]=geometry.bbox,s=Math.min(1024*.88/(x1-x0),768*.46/(y1-y0)),tx=512-s*(x0+x1)/2,ty=768*.8-s*geometry.ground_y,m=[s,0,tx,0,s,ty],a=Float32Array.from(alpha.data,v=>v/255),premult=Float32Array.from(rgb.data,(v,i)=>v*a[Math.floor(i/3)]);
  const placedA=warp(raster(rgb.w,rgb.h,1,a),m,1024,768),placedP=warp(raster(rgb.w,rgb.h,3,premult),m,1024,768);placedA.data=Float32Array.from(placedA.data,v=>clip(v));
  const color=raster(1024,768,3,Uint8Array.from(placedP.data,(v,i)=>placedA.data[Math.floor(i/3)]>1e-6?u8(v/placedA.data[Math.floor(i/3)]):0)),pa=raster(1024,768,1,Uint8Array.from(placedA.data,v=>u8(v*255))),g=structuredClone(geometry),point=p=>[s*p[0]+tx,s*p[1]+ty];g.bbox=[...point([x0,y0]),...point([x1,y1])];
  for(const key of ['footprint','chassis_footprint','lower_body_profile'])g[key]=g[key].map(point);g.ground_y=s*g.ground_y+ty;g.ground_anchor=point(g.ground_anchor);
  for(const c of [...g.contacts,...g.inferred_contacts]){[c.x,c.y]=point([c.x,c.y]);c.radius*=s;c.half_width*=s;if(c.ellipse){const e=c.ellipse;[e.cx,e.cy]=point([e.cx,e.cy]);e.rx*=s;e.ry*=s;}}
  g.view_cues.opposite_track_shift=g.view_cues.opposite_track_shift.map(v=>v*s);return {rgb:color,alpha:pa,geometry:g,matrix:m};
}
function contrast(rgb,factor){ // Pillow ImageEnhance.Contrast: rounded L mean, truncating blend.
  let total=0;for(let i=0;i<rgb.data.length;i+=3)total+=Math.floor((19595*rgb.data[i]+38470*rgb.data[i+1]+7471*rgb.data[i+2]+32768)/65536);const level=Math.floor(total/(rgb.w*rgb.h)+.5);return raster(rgb.w,rgb.h,3,Uint8Array.from(rgb.data,v=>clip(Math.trunc(level+(v-level)*factor),0,255)));
}
export function transparent(rgb,alpha,shadow,enhancement){if(enhancement)rgb=contrast(rgb,1.02);const out=new Uint8Array(rgb.w*rgb.h*4);for(let i=0;i<alpha.data.length;i++){const a=alpha.data[i]/255,combined=a+shadow[i]*(1-a);for(let c=0;c<3;c++)out[i*4+c]=combined>1e-6?u8(rgb.data[i*3+c]*a/combined):0;out[i*4+3]=u8(combined*255);}return raster(rgb.w,rgb.h,4,out);}
function samples(im,x0,y0,x1,y1,lo,hi,spread){const list=[];for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const i=(y*im.w+x)*3,p=Array.from(im.data.subarray(i,i+3),v=>v/255),m=mean(p);if(m>lo&&m<hi&&Math.max(...p)-Math.min(...p)<spread)list.push(p);}return list;}
const medianRGB=p=>[0,1,2].map(c=>median(p.map(v=>v[c])));
function matchLight(colors,alpha,bg){const small=resize(bg,200,120,cv.INTER_LINEAR),neutral=samples(small,30,72,170,108,.18,.8,.18),paint=[];for(let i=0;i<alpha.length;i++){const p=Array.from(colors.subarray(i*3,i*3+3));if(alpha[i]>.98&&mean(p)>.65&&Math.max(...p)-Math.min(...p)<.18)paint.push(p);}let gain=[1,1,1],exposure=1;if(paint.length>100&&neutral.length>100){const source=medianRGB(paint),ambient=medianRGB(neutral),am=mean(ambient),sm=mean(source);gain=ambient.map((v,c)=>clip(((.6*v/am+.4)/(source[c]/sm))**.65,.94,1.06));exposure=clip(.98+.12*(am-.5),.95,1.02);}return Float32Array.from(colors,(v,i)=>clip(v*gain[i%3]*exposure));}
function ambient(colors,alpha,srcW,srcH,bg,g,m){
  const scene=resize(bg,240,160),[x0,y0,x1,y1]=g.bbox;
  const chroma=(pixels,strength,limit)=>{if(pixels.length<24)return [1,1,1];const v=medianRGB(pixels),avg=mean(v);return v.map(c=>clip(1+strength*(c/avg-1),1-limit,1+limit));};
  const sky=chroma(samples(scene,24,0,216,64,.15,.92,.25),.2,.025),gy=Math.trunc(clip((g.ground_y*m[0]+m[5])/bg.h*160,0,159)),roads=[];
  for(const f of [.15,.5,.85]){const gx=Math.trunc(clip(((x0+(x1-x0)*f)*m[0]+m[2])/bg.w*240,0,239));roads.push(chroma(samples(scene,Math.max(0,gx-18),Math.max(0,gy-4),Math.min(240,gx+18),Math.min(160,gy+18),.15,.92,.25),.35,.045));}
  const out=new Float32Array(colors.length);for(let y=0;y<srcH;y++){let lower=clip((clip((y-y0)/Math.max(1,y1-y0))-.48)/.48);lower=lower*lower*(3-2*lower);for(let x=0;x<srcW;x++){const i=y*srcW+x,p=colors.subarray(i*3,i*3+3),lum=p[0]*.2126+p[1]*.7152+p[2]*.0722,saturation=(Math.max(...p)-Math.min(...p))/Math.max(Math.max(...p),.05),protect=1-.75*clip(saturation),xf=clip((x-x0)/Math.max(1,x1-x0));for(let c=0;c<3;c++){const ground=interp(xf,[.15,.5,.85],roads.map(r=>r[c])),gain=1+protect*((sky[c]-1)*(1-lower)*.65+(ground-1)*lower),bounce=.025*lower*lum*(1-lum);out[i*3+c]=alpha[i]>0?clip(p[c]*gain+bounce*protect*ground):p[c];}}}return out;
}
export function parking(rgb,alpha,shadow,g,bg,enhancement){
  const [x0,y0,x1,y1]=g.bbox,s=Math.min(.88*bg.w/(x1-x0),.48*bg.h/(y1-y0)),tx=.5*bg.w-s*(x0+x1)/2,ty=clip(.55*bg.h-s*(y0+y1)/2,6-s*y0,bg.h-6-s*y1),m=[s,0,tx,0,s,ty],a=Float32Array.from(alpha.data,v=>v/255);let colors=Float32Array.from(rgb.data,v=>v/255);
  if(enhancement){colors=matchLight(colors,a,bg);colors=ambient(colors,a,rgb.w,rgb.h,bg,g,m);}
  const wa=warp(raster(rgb.w,rgb.h,1,a),m,bg.w,bg.h).data,wp=warp(raster(rgb.w,rgb.h,3,Float32Array.from(colors,(v,i)=>v*a[Math.floor(i/3)])),m,bg.w,bg.h).data,ws=warp(raster(rgb.w,rgb.h,1,Float32Array.from(shadow,v=>u8(v*255)/255)),m,bg.w,bg.h).data;
  let combined=raster(bg.w,bg.h,3,Uint8Array.from(wp,(v,i)=>{const j=Math.floor(i/3),alpha=clip(wa[j]);return u8((clip(v,0,alpha)+bg.data[i]/255*(1-clip(ws[j]))*(1-alpha))*255);}));if(enhancement)combined=contrast(combined,1.02);return combined;
}
