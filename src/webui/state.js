export const pages = {
  models: ['モデル', 'ローカルモデルと現在の稼働状態を管理します。'],
  catalog: ['カタログ・取得', 'モデルの選択から取得状況の確認まで。'],
  playground: ['Playground', 'ロード済みモデルの出力を、小さなテキストで確かめます。'],
  activity: ['推論ログ', '処理時間と成功・失敗を確認します。'],
  settings: ['稼働設定', '現在の実行環境と起動設定を確認します。'],
};

export function formatBytes(value) {
  if (value == null || !Number.isFinite(Number(value))) return '不明';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let n = Math.max(0, Number(value)), i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function formatTime(timestamp) {
  return timestamp ? new Date(timestamp * 1000).toLocaleString('ja-JP') : '—';
}

export function modelState(model, status, jobs) {
  const job = jobs.find(j => j.local_name === model.id);
  if (job && ['queued', 'downloading', 'failed'].includes(job.status)) return job.status === 'failed'
    ? ['取得失敗・部分ファイルの可能性', 'error'] : ['取得中・ロード不可', 'warn'];
  const entry = status?.loaded?.find(m => m.id === model.id && m.type === model.type);
  if (model.loaded) return entry?.last_inference_at
    ? ['推論確認済み（現ロード）', 'good'] : ['ロード済み・推論未確認', 'warn'];
  return ['ローカル検出・互換性未確認', ''];
}

export function catalogLocalName(item) {
  return item.local_name || item.repo_id.split('/').at(-1);
}

export function validateDownload(repoId, localName) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repoId) || repoId.split('/').some(p => p === '.' || p === '..'))
    return 'リポジトリ ID は owner/repo 形式で入力してください。';
  if (localName && (localName.startsWith('/') || localName.includes('\\') || /^[A-Za-z]:/.test(localName)
      || localName.split('/').some(p => !p || p === '.' || p === '..')))
    return '保存先は MODEL_DIR 配下の相対パスにしてください。空の階層・「.」「..」は使えません。';
  return '';
}

export function testPayload(model, a, b, dimensions) {
  if (!model) throw new Error('ロード済みの embedding モデルを選択してください。');
  const inputs = [a.trim(), b.trim()];
  if (inputs.some(t => !t || t.length > 2000)) throw new Error('テキストはそれぞれ 1〜2,000 文字で入力してください。');
  const body = {model, input: inputs, encoding_format: 'float'};
  if (dimensions !== '') {
    const n = Number(dimensions);
    if (!Number.isSafeInteger(n) || n < 1) throw new Error('出力次元は 1 以上の整数にしてください。');
    body.dimensions = n;
  }
  return body;
}

export function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || a.length !== b.length
      || [...a, ...b].some(n => typeof n !== 'number' || !Number.isFinite(n)))
    throw new Error('サーバーから有効なベクトルを取得できませんでした。');
  const normA = Math.hypot(...a), normB = Math.hypot(...b);
  if (!normA || !normB) return null;
  return Math.max(-1, Math.min(1, a.reduce((sum, v, i) => sum + v * b[i], 0) / (normA * normB)));
}
