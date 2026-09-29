// Every temporary OpenCV/WASM allocation is explicitly released.
export const raster=(w,h,c=1,data=null)=>({w,h,c,data:data??new Float32Array(w*h*c)});
function mat(im){const cv=globalThis.cv;return cv.matFromArray(im.h,im.w,cv[`${im.data instanceof Uint8Array?'CV_8UC':'CV_32FC'}${im.c}`],im.data);}
function result(m){const byte=m.depth()===globalThis.cv.CV_8U;return raster(m.cols,m.rows,m.channels(),byte?new Uint8Array(m.data):new Float32Array(m.data32F));}
export function op(im,fn){const a=mat(im),b=new cv.Mat();try{fn(a,b);return result(b);}finally{a.delete();b.delete();}}
export const resize=(im,w,h,mode=cv.INTER_AREA)=>op(im,(a,b)=>cv.resize(a,b,new cv.Size(w,h),0,0,mode));
export const resizeScale=(im,sx,sy)=>op(im,(a,b)=>cv.resize(a,b,new cv.Size(0,0),sx,sy,cv.INTER_AREA));
export const gray=im=>op(im,(a,b)=>cv.cvtColor(a,b,cv.COLOR_RGB2GRAY));
export function crop(im,x,y,w,h){const data=new im.data.constructor(w*h*im.c);for(let j=0;j<h;j++)data.set(im.data.subarray(((y+j)*im.w+x)*im.c,((y+j)*im.w+x+w)*im.c),j*w*im.c);return raster(w,h,im.c,data);}
export const blur=(im,sx,sy=sx,border=cv.BORDER_DEFAULT,k=0)=>op(im,(a,b)=>cv.GaussianBlur(a,b,new cv.Size(k,k),sx,sy,border));
export function morph(im,kind,size=5,ellipse=false){const kernel=cv.getStructuringElement(ellipse?cv.MORPH_ELLIPSE:cv.MORPH_RECT,new cv.Size(size,size));try{return op(im,(a,b)=>cv.morphologyEx(a,b,kind,kernel));}finally{kernel.delete();}}
export function components(im){const a=mat(im),labels=new cv.Mat(),stats=new cv.Mat(),centroids=new cv.Mat();try{const n=cv.connectedComponentsWithStats(a,labels,stats,centroids,8,cv.CV_32S);return {labels:new Int32Array(labels.data32S),stats:Array.from({length:n},(_,i)=>Array.from(stats.data32S.slice(i*5,i*5+5)))};}finally{[a,labels,stats,centroids].forEach(x=>x.delete());}}
export function largest(alpha,threshold=128){const bin=raster(alpha.w,alpha.h,1,Uint8Array.from(alpha.data,v=>v>=threshold?1:0)),cc=components(bin);if(cc.stats.length<2)throw Error('No foreground vehicle found.');let id=1;for(let i=2;i<cc.stats.length;i++)if(cc.stats[i][4]>cc.stats[id][4])id=i;const [x,y,w,h,area]=cc.stats[id];return {mask:raster(alpha.w,alpha.h,1,Uint8Array.from(cc.labels,v=>v===id?1:0)),bbox:[x,y,x+w,y+h],area};}
export function warp(im,m,w,h,mode=cv.INTER_LANCZOS4){const M=cv.matFromArray(2,3,cv.CV_32F,m);try{return op(im,(a,b)=>cv.warpAffine(a,b,M,new cv.Size(w,h),mode,cv.BORDER_CONSTANT,new cv.Scalar()));}finally{M.delete();}}
export function polygon(points,w,h){const n=256,d=new Float32Array(n*n);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const u=(x-127.5)/127.5,v=(y-127.5)/127.5,r=(Math.abs(u)**3.2+Math.abs(v)**3.2)**(1/3.2),t=Math.max(0,Math.min(1,(r-.64)/.36));d[y*n+x]=1-t*t*(3-2*t);}const a=cv.matFromArray(4,1,cv.CV_32FC2,[0,255,255,255,255,0,0,0]),b=cv.matFromArray(4,1,cv.CV_32FC2,points.flat()),M=cv.getPerspectiveTransform(a,b);try{return op(raster(n,n,1,d),(src,dst)=>cv.warpPerspective(src,dst,M,new cv.Size(w,h),cv.INTER_LINEAR,cv.BORDER_CONSTANT,new cv.Scalar()));}finally{a.delete();b.delete();M.delete();}}
export function circles(im,bh,sy){const a=mat(im),out=new cv.Mat();try{cv.HoughCircles(a,out,cv.HOUGH_GRADIENT,1.3,Math.max(12,bh*sy*.14),85,24,Math.max(7,Math.trunc(bh*sy*.10)),Math.max(9,Math.trunc(bh*sy*.28)));const d=out.data32F;return Array.from({length:Math.min(100,d.length/3)},(_,i)=>Array.from(d.slice(i*3,i*3+3)));}finally{a.delete();out.delete();}}
export function sample(im,x,y){ // OpenCV INTER_LINEAR quantizes coordinates to 1/32.
  x=Math.round(x*32)/32;y=Math.round(y*32)/32;const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;
  const at=(xx,yy)=>xx<0||yy<0||xx>=im.w||yy>=im.h?255:im.data[yy*im.w+xx];
  return Math.round((at(ix,iy)*(1-fx)+at(ix+1,iy)*fx)*(1-fy)+(at(ix,iy+1)*(1-fx)+at(ix+1,iy+1)*fx)*fy);
}
export function median7(im){const out=new Float32Array(im.data.length),v=new Float32Array(49);for(let y=0;y<im.h;y++)for(let x=0;x<im.w;x++){let n=0;for(let j=-3;j<=3;j++)for(let i=-3;i<=3;i++)v[n++]=im.data[Math.max(0,Math.min(im.h-1,y+j))*im.w+Math.max(0,Math.min(im.w-1,x+i))];v.sort();out[y*im.w+x]=v[24];}return raster(im.w,im.h,1,out);}
