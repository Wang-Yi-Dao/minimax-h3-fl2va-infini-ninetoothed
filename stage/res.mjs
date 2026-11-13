import { api } from './gpu.mjs';
const r = await api('/api/resources');
const j = r.json;
for (const p of (j.pools||[])) {
  console.log('POOL', p.id, 'policy', JSON.stringify(p.policy));
  console.log('EXP', JSON.stringify(p.expansion_policy));
  for (const n of (p.nodes||[])) {
    console.log('  node', n.id, 'cards', n.cards, 'free', n.free, 'sched', n.sched, 'act', n.act, 'ready', n.ready);
  }
}
console.log('ALLOC', JSON.stringify(j.allocations||[]).slice(0,600));
console.log('GRANTS', JSON.stringify(j.grants||[]).slice(0,600));
