import {clip,mean,median,quantile,round,smooth1d,median1d,peaks} from './numeric.js';
import {raster,largest,morph,gray,crop,blur,resizeScale,circles,sample} from './cv.js';

export function refineMask(alpha){const {mask,area}=largest(alpha);if(area<alpha.w*alpha.h*.01)throw Error('Foreground is too small to process reliably.');const support=morph(mask,cv.MORPH_DILATE,9);return raster(alpha.w,alpha.h,1,Uint8Array.from(alpha.data,(v,i)=>!support.data[i]||v<3?0:v>252?255:v));}

function ellipseScore(gr,mask,bottom,bbox,cx,cy,rx,ry){
  const [x0,y0,x1,y1]=bbox,bh=y1-y0,bw=x1-x0;
  if(!(cx>x0+.018*bw&&cx<x1-.018*bw&&cy>y0+.51*bh&&cy<y1-.035*bh&&ry>.105*bh&&ry<.315*bh))return null;
  const bx=clip(round(cx)-x0,0,bottom.length-1),error=Math.abs(cy+ry-bottom[bx]);if(error>.11*bh)return null;
  const ring=[],fg=[];for(const level of [.81,.9,.98])for(let i=0;i<72;i++){const a=i*2*Math.PI/72,x=cx+rx*level*Math.cos(a),y=cy+ry*level*Math.sin(a);ring.push(sample(gr,x,y));fg.push(sample(mask,x,y)/255);}
  if(mean(fg)<.88)return null;
  const rubber=mean(ring.map(v=>v<86?1:0)),ringMean=mean(ring);if(rubber<.57||ringMean>96)return null;
  const inside=[];for(let j=0;j<8;j++)for(let i=0;i<72;i++){const level=.12+j*(.68-.12)/7,a=i*2*Math.PI/72;inside.push(sample(gr,cx+rx*level*Math.cos(a),cy+ry*level*Math.sin(a)));}
  const texture=clip((quantile(inside,.88)-quantile(inside,.12)-15)/75),spokes=clip((quantile(inside,.85)-ringMean-5)/55),contact=Math.exp(-.5*(error/Math.max(3,.037*bh))**2),spread=Math.max(2,Math.trunc(rx*.85));
  const protrusion=clip((bottom[bx]-(bottom[clip(bx-spread,0,bottom.length-1)]+bottom[clip(bx+spread,0,bottom.length-1)])/2)/Math.max(3,.34*ry));
  let score=.32*rubber+.23*contact+.14*texture+.12*spokes+.19*protrusion;if(contact<.14||(texture<.12&&protrusion<.2))score*=.62;
  return {score,evidence:{rubber,rim_detail:texture,contact_agreement:contact,protrusion}};
}

function wheelEllipses(rgb,mask,bbox,bottom,expanded=false){
  const [x0,y0,x1,y1]=bbox,bw=x1-x0,bh=y1-y0,gr=gray(rgb),cropY=Math.trunc(y0+.48*bh),roi=blur(crop(gr,x0,cropY,bw,y1-cropY),1,1,cv.BORDER_DEFAULT,5),scale=Math.min(1,760/bw),proposals=[];
  const mask255=raster(mask.w,mask.h,1,Uint8Array.from(mask.data,v=>v*255));
  for(const peak of peaks(bottom,Math.max(4,.02*bh),Math.max(12,Math.trunc(.12*bw))))for(const shift of [-.035,0,.035])for(const rf of expanded?[.11,.13,.15,.18,.21,.24]:[.15,.18,.21,.24])for(const oval of [.48,.62,.76,.94]){const ry=rf*bh;proposals.push([x0+peak+shift*bh,bottom[peak]-ry,ry*oval,ry]);}
  for(const ratio of [.48,.62,.78,.96]){const sx=scale/ratio,sy=scale,stretched=resizeScale(roi,sx,sy);for(const [cx,cy,r] of circles(stretched,bh,sy))for(const expansion of [1,1.12,1.25,1.38])proposals.push([x0+cx/sx,cropY+cy/sy,r/sx*expansion,r/sy*expansion]);}
  const ranked=[];for(const [cx,cy,rx,ry] of proposals){const scored=ellipseScore(gr,mask255,bottom,bbox,cx,cy,rx,ry);if(!scored)continue;let score=scored.score+.065*clip((ry/bh-.12)/.10);if(expanded&&scored.evidence.protrusion<.15)score*=.62;if(score>=.63)ranked.push({cx,cy,rx,ry,score,evidence:scored.evidence});}
  ranked.sort((a,b)=>b.score-a.score);const candidates=[];for(const c of ranked){if(!candidates.some(o=>Math.abs(c.cx-o.cx)<Math.max(c.rx,o.rx)*.18&&Math.abs(c.cy-o.cy)<Math.max(c.ry,o.ry)*.18&&Math.abs(c.rx/c.ry-o.rx/o.ry)<.08))candidates.push(c);if(candidates.length>=90)break;}
  if(candidates.length<2)return candidates;
  let selected=candidates.slice(0,1),best=-Infinity;for(let i=0;i<candidates.length;i++)for(let j=i+1;j<candidates.length;j++){const a=candidates[i],b=candidates[j],span=Math.abs(a.cx-b.cx),ratio=Math.min(a.ry,b.ry)/Math.max(a.ry,b.ry),ovalA=a.rx/a.ry,ovalB=b.rx/b.ry;if(span<.28*bw||ratio<(expanded?.40:.48)||Math.abs(ovalA-ovalB)>.34)continue;const score=a.score+b.score+.13*Math.min(span/(.65*bw),1)-.8*Math.abs(ovalA-ovalB);if(score>best){best=score;selected=[a,b];}}
  if(!expanded&&(selected.length<2||selected.some(c=>c.evidence.protrusion<.15))){const refined=wheelEllipses(rgb,mask,bbox,bottom,true);if(refined.length===2)return refined;}
  return selected;
}

function fallback(rgb,bbox,bottom){const [x0,y0,x1,y1]=bbox,bw=x1-x0,bh=y1-y0,gr=gray(rgb),contacts=[];
  for(const [lo,hi] of [[.035,.26],[.74,.965]]){const xs=Array.from({length:Math.trunc(x0+hi*bw)-Math.trunc(x0+lo*bw)},(_,i)=>Math.trunc(x0+lo*bw)+i),ys=xs.map(x=>bottom[clip(x-x0,0,bottom.length-1)]);let dark=xs.map((x,i)=>{let sum=0,n=0;for(let y=Math.max(y0,Math.trunc(ys[i]-.06*bh));y<Math.trunc(ys[i])+1;y++)for(let xx=Math.max(x0,x-2);xx<Math.min(x1,x+3);xx++){sum+=gr.data[y*gr.w+xx];n++;}return n?1-sum/n/255:0;});dark=smooth1d(dark,Math.max(1,.008*bw));const q=quantile(bottom,.45),scores=ys.map((y,i)=>(y-q)/Math.max(1,.07*bh)+.48*dark[i]);let peak=0;for(let i=1;i<scores.length;i++)if(scores[i]>scores[peak])peak=i;const good=scores.map((s,i)=>i).filter(i=>scores[i]>=scores[peak]-.12&&Math.abs(i-peak)<.055*bw),ix=good.length?Math.trunc(median(good)):peak,radius=clip(.145*bh,.035*bw,.11*bw);contacts.push({x:xs[ix],y:ys[ix],radius,half_width:Math.max(.025*bw,.35*radius),confidence:.39+.16*dark[ix],source:'occluded_tire_lower_silhouette'});}
  return contacts;
}

export function estimateGeometry(rgb,alpha){
  const {mask,bbox}=largest(alpha),[x0,y0,x1,y1]=bbox,bw=x1-x0,bh=y1-y0;if(Math.min(bw,bh)<24)throw Error('Vehicle foreground too small for geometry.');
  const raw=Float64Array.from({length:bw},(_,i)=>{for(let y=y1-1;y>=y0;y--)if(mask.data[y*mask.w+x0+i])return y;return y1-1;});
  const bottom=smooth1d(median1d(raw,5),1.2),wheels=bw/bh>1.75?wheelEllipses(rgb,mask,bbox,bottom).sort((a,b)=>a.cx-b.cx):[];
  let contacts=wheels.map(w=>{const ix=clip(round(w.cx)-x0,0,bottom.length-1),half=Math.max(2,Math.trunc(.24*w.rx));return {x:w.cx,y:quantile(bottom.slice(Math.max(0,ix-half),Math.min(bottom.length,ix+half+1)),.9),radius:w.ry,half_width:w.rx*.32,confidence:Math.min(.99,w.score),source:'rubber_rim_ellipse',ellipse:{cx:w.cx,cy:w.cy,rx:w.rx,ry:w.ry},evidence:w.evidence};});
  const inferred=[];let footprint,shiftX=0,shiftY,asym=0,oval=0,perspective=0,side=0,mode;
  if(contacts.length===2){const [left,right]=contacts,lg=left.x-x0-wheels[0].rx,rg=x1-right.x-wheels[1].rx;asym=(lg-rg)/bw;oval=mean(wheels.map(w=>w.rx/w.ry));shiftX=clip(-.85*(lg-rg),-.42*bw,.42*bw);if(Math.abs(asym)<.065)shiftX*=.35;shiftY=-(.105*bh+Math.abs(shiftX)*.10);for(const c of contacts)inferred.push({x:c.x+shiftX,y:c.y+shiftY,radius:c.radius*.9,half_width:c.half_width,confidence:.34,source:'inferred_opposite_track'});footprint=[[left.x,left.y],[right.x,right.y],[inferred[1].x,inferred[1].y],[inferred[0].x,inferred[0].y]];perspective=clip(Math.abs(shiftX)/(.3*bw));side=clip((oval-.5)/.48)*(1-.6*perspective);mode='visible_wheel_pair';}
  else{contacts=fallback(rgb,bbox,bottom);const [left,right]=contacts,width=right.x-left.x,depth=.12*bh;footprint=[[left.x,left.y],[right.x,right.y],[right.x-.11*width,right.y-depth],[left.x+.11*width,left.y-depth]];shiftY=-depth;mode='occluded_axle_pair';}
  const support=footprint.map(p=>p.slice());if(mode==='visible_wheel_pair'){const along=[support[1][0]-support[0][0],support[1][1]-support[0][1]];for(const i of [0,3])for(let j=0;j<2;j++)support[i][j]-=.27*along[j];for(const i of [1,2])for(let j=0;j<2;j++)support[i][j]+=.27*along[j];}else{for(const i of [0,3])support[i][0]-=.035*bw;for(const i of [1,2])support[i][0]+=.035*bw;}
  const ground=Math.max(...contacts.map(c=>c.y)),radius=median(contacts.map(c=>c.radius));support.forEach((p,i)=>{p[0]=clip(p[0],x0+.025*bw,x1-.025*bw);if(i<2)p[1]=Math.min(p[1],ground+.06*radius);});
  const indices=[...new Set(Array.from({length:Math.min(128,bw)},(_,i)=>Math.trunc(i*(bw-1)/(Math.min(128,bw)-1))))];
  return {bbox,contacts,inferred_contacts:inferred,lower_body_profile:indices.map(i=>[x0+i,bottom[i]]),footprint,chassis_footprint:support,ground_y:ground,ground_anchor:[mean(contacts.map(c=>c.x)),ground],confidence:mean(contacts.map(c=>c.confidence)),view_cues:{support_mode:mode,detected_wheel_count:wheels.length,aspect_ratio:bw/bh,wheel_ellipse_ratio:oval,side_on_score:side,perspective_score:perspective,overhang_asymmetry:asym,opposite_track_shift:[shiftX,shiftY],contact_slope:(contacts.at(-1).y-contacts[0].y)/Math.max(1,contacts.at(-1).x-contacts[0].x)}};
}
