// submit a job against an EXISTING snapshot (no saveText/snapshot): SNAP=... NAME=... CNT=8 TMO=1300 [ARGS="2 1200"]
import { api } from './gpu.mjs';
const P='0218515d7d5b40d0b710ffa6cc8a2690';
const POOL='scnet-dcu';
const ENV='524820f26b724c00add8d17183cfad10e8572f048a29707542745eedb42a3aad';
const SNAP=process.env.SNAP, NAME=process.env.NAME||'snapjob';
const CNT=+(process.env.CNT||8), TMO=+(process.env.TMO||1300);
const ARGS=process.env.ARGS? process.env.ARGS.split(' ').filter(Boolean):[];
const rid=()=>Math.random().toString(16).slice(2).padEnd(32,'0').slice(0,32);
if(!SNAP){ console.log('NO_SNAP'); process.exit(1); }
const cfg={entrypoint:'cmp1.sh',args:ARGS,kind:'batch'};
const c=await api('/api/projects/config',{project:P,revision:undefined,config:cfg},60000);
console.log('cfg',c.status,c.text.slice(0,120));
const pf=await api('/api/batch/preflight',{project:P,snapshot:SNAP,pool:POOL,timeout:TMO,environment_digest:ENV,kind:'batch',resources:{scope:'unit',count:CNT}},120000);
console.log('pf',pf.status,pf.text.slice(0,300));
if(pf.status!==200||!(pf.json&&pf.json.allowed!==false)){ console.log('PF_BLOCKED'); process.exit(2); }
const s=await api('/api/batch/submit',{project:P,pool:POOL,timeout:TMO,name:NAME,environment_digest:ENV,snapshot:SNAP,request_id:rid(),resources:{scope:'unit',count:CNT}},180000);
console.log('submit',s.status,JSON.stringify(s.json).slice(0,300));
