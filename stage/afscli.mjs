import * as afs from './afs.mjs';
const [,,cmd,...args]=process.argv;
if(cmd==='ls'){
  const p=args[0]||'';
  const es=await afs.list(p);
  for(const e of es) console.log((e.directory?'d':e.symlink?'l':'-'), String(e.bytes).padStart(10), (e.name||'').padEnd(36), (e.version||'').split(':')[1]||'');
  console.log('N='+es.length);
} else if(cmd==='dl'){
  const r=await afs.download(args[0],args[1]);
  console.log('DL',args[0],'bytes',r.size);
} else if(cmd==='up'){
  const r=await afs.upload(args[0],args[1]);
  console.log('UP',args[0],'->',args[1],'bytes',r.bytes,'sha',r.sha.slice(0,16));
} else console.log('usage: node afscli.mjs ls [path] | dl <remote> <local> | up <local> <remote>');
