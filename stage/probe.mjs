import { api } from './gpu.mjs';
const h=await api('/api/health'); const hj=h.json||{};
const r=await api('/api/resources');
if(r.status===401){ console.log('PLATFORM ready='+hj.ready+' AUTH_REQUIRED token 过期 VERDICT=NOAUTH'); }
else { const rj=r.json||{}; const pool=(rj.pools||[])[0]||{}; const nodes=pool.nodes||[];
 let free=0, ready=0, certFree=0, certReady=0, uncert=0; const imgs=new Set();
 for(const n of nodes){ const t=n.terminal||{}; const certified = t.gpu===true && !t.gpu_reason;
   if(n.ready){ ready++; if(n.image) imgs.add(n.image);
     if(certified){ certReady++; certFree += (n.free||0); } else if(t.gpu===true) uncert++; }
   free += (n.free||0); }
 const ok = hj.ready===true && certFree>0;
 console.log('PLATFORM ready='+hj.ready+' nodes_ready='+ready+'/'+nodes.length+' free='+free+' CERTIFIED_nodes='+certReady+' certified_free='+certFree+' uncertified_gpu_nodes='+uncert+' images='+[...imgs].map(x=>x.slice(0,19)).join(',')+' VERDICT='+(ok?'USABLE':'UNAVAILABLE')); }