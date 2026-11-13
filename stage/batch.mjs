import { api } from './gpu.mjs';

export const PROJECT = process.env.PROJECT_ID || '0218515d7d5b40d0b710ffa6cc8a2690';
export const POOL = 'scnet-dcu';
export const ENV_DIGEST = '524820f26b724c00add8d17183cfad10e8572f048a29707542745eedb42a3aad';
const rid = () => Math.random().toString(16).slice(2).padEnd(32, '0').slice(0, 32);

export async function project() {
  const r = await api('/api/projects?id=' + PROJECT);
  return r.json;
}

export async function saveText(path, text, expected) {
  const body = { project: PROJECT, path, text, expected_revision: expected, request_id: rid() };
  const r = await api('/api/projects/text', body, 90000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 300) };
}

export async function saveConfig(config, revision) {
  const r = await api('/api/projects/config', { project: PROJECT, revision, config }, 60000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 300) };
}

export async function snapshot() {
  const r = await api('/api/projects/snapshot', { project: PROJECT }, 240000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 300) };
}

export async function preflight(snap, timeout, count = 1) {
  const r = await api('/api/batch/preflight', { project: PROJECT, snapshot: snap, pool: POOL, timeout, environment_digest: ENV_DIGEST, kind: 'batch', resources: { scope: 'unit', count } }, 120000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 500) };
}

export async function submit(snap, timeout, name, count = 1) {
  // NOTE: submit must NOT carry `kind` (403 INVALID_BATCH_SUBMISSION); only preflight takes kind.
  const r = await api('/api/batch/submit', { project: PROJECT, pool: POOL, timeout, name, environment_digest: ENV_DIGEST, snapshot: snap, request_id: rid(), resources: { scope: 'unit', count } }, 180000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 500) };
}

export async function jobs() {
  const r = await api('/api/batch', undefined, 60000);
  return { status: r.status, json: r.json };
}

export async function job(id) {
  const r = await api('/api/batch?id=' + encodeURIComponent(id), undefined, 60000);
  const list = (r.json && (r.json.jobs || (r.json.job ? [r.json.job] : []))) || [];
  const j = list.find(x => x && x.id === id) || list[0] || null;
  return { status: r.status, job: j, raw: r.json };
}

export async function cancel(id) {
  const r = await api('/api/batch/cancel', { id }, 60000);
  return { status: r.status, json: r.json, text: r.text.slice(0, 300) };
}
