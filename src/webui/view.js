import {formatBytes, formatTime, modelState, catalogLocalName, cosineSimilarity} from './state.js';
export const $ = id => document.getElementById(id);
export function el(tag, className = '', text = '') {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
const badge = (text, kind = '') => el('span', `badge ${kind}`, text);
const empty = text => el('div', 'empty', text);
const action = (label, handler, className = 'btn small', disabled = false) => {
  const button = el('button', className, label); button.type = 'button'; button.disabled = disabled;
  button.addEventListener('click', handler); return button;
};
const metric = (label, value, detail) => {
  const node = el('div', 'metric'); node.append(el('div', 'metric-label', label), el('div', 'metric-value', value), el('div', 'metric-detail', detail)); return node;
};

export function renderRuntime(state) {
  const target = $('runtime');
  if (!state.health) { target.replaceChildren(empty('稼働状態を取得できません。状態を更新してください。')); return; }
  const h = state.health, gpu = h.gpu, count = state.models.filter(m => m.loaded).length;
  const device = metric('実行デバイス', (h.accelerator_backend || h.device).toUpperCase(), gpu?.name || 'CPU 実行');
  const loaded = metric('ロード済みモデル', String(count), `ローカル一覧 ${state.models.length} 件 · アイドル TTL ${h.idle_ttl || '無効'}`);
  const memory = metric(gpu ? 'GPU メモリ（全プロセス）' : 'GPU メモリ', gpu ? `${(gpu.vram_free_mb / 1024).toFixed(1)} GiB 空き` : '使用なし', gpu ? `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GiB 使用` : 'CPU のメモリ使用量は未提供');
  if (gpu) { const bar = el('div', 'resource'), fill = el('span'); fill.style.width = `${Math.min(100, Math.max(0, gpu.vram_used_mb / gpu.vram_total_mb * 100))}%`; bar.append(fill); memory.append(bar); }
  target.replaceChildren(device, loaded, memory);
}

export function renderModels(state, handlers) {
  const query = $('model-search').value.toLowerCase();
  const models = state.models.filter(m => m.id.toLowerCase().includes(query));
  $('model-count').textContent = state.models.length;
  const signature = JSON.stringify([models, state.status?.loaded, state.jobs.map(j => [j.local_name, j.status]), state.busy, query]);
  if ($('models').dataset.signature === signature) return;
  $('models').dataset.signature = signature;
  const selected = new Map([...$('models').querySelectorAll('select')].map(n => [n.closest('.model-row').dataset.id, n.value]));
  const focused = document.activeElement;
  const focusedId = focused?.closest('.model-row')?.dataset.id;
  const focusedLabel = focused?.tagName === 'SELECT' ? 'select' : focused?.textContent;
  if (!models.length) { $('models').replaceChildren(empty(query ? '検索条件に一致するモデルがありません。' : 'ローカルモデルがありません。カタログから取得できます。')); return; }
  const rows = models.map(model => {
    const row = el('div', 'model-row'), content = el('div');
    row.dataset.id = model.id;
    content.append(el('div', 'model-title mono', model.id));
    const badges = el('div', 'model-badges'), [label, kind] = modelState(model, state.status, state.jobs);
    badges.append(badge(model.type === 'unknown' ? '種別未判定' : model.type), badge(label, kind));
    content.append(badges, el('div', 'muted', model.loaded ? `最終使用 ${formatTime(model.last_used)}` : 'モデル形式・依存環境はロード時に確認されます。'));
    const controls = el('div', 'model-actions');
    if (model.loaded) {
      if (model.type === 'embedding') controls.append(action('試す', () => handlers.test(model.id)));
      controls.append(action('アンロード', () => handlers.unload(model.id), 'btn small danger', state.busy));
    } else {
      const select = el('select'); select.setAttribute('aria-label', `${model.id} のロード種別`);
      for (const type of ['embedding', 'reranker']) { const option = el('option', '', type); option.value = type; select.append(option); }
      if (model.id.toLowerCase().includes('rerank')) select.value = 'reranker';
      if (selected.has(model.id)) select.value = selected.get(model.id);
      select.disabled = state.busy;
      const latestJob = state.jobs.find(j => j.local_name === model.id);
      const blocked = latestJob && ['queued', 'downloading', 'failed'].includes(latestJob.status);
      controls.append(select, action('ロード', () => handlers.load(model.id, select.value), 'btn small primary', state.busy || blocked));
    }
    row.append(content, controls); return row;
  });
  $('models').replaceChildren(...rows);
  if (focusedId) {
    const row = rows.find(n => n.dataset.id === focusedId);
    const target = focusedLabel === 'select' ? row?.querySelector('select') : [...(row?.querySelectorAll('button') || [])].find(n => n.textContent === focusedLabel);
    if (target && !target.disabled) target.focus({preventScroll: true});
  }
}

export function renderCatalog(state, handlers) {
  const signature = JSON.stringify([state.catalog, state.models.map(m => [m.id, m.loaded]), state.jobs.map(j => [j.repo_id, j.local_name, j.status]), state.busy]);
  if ($('catalog').dataset.signature === signature) return;
  $('catalog').dataset.signature = signature;
  if (!state.catalog.length) { $('catalog').replaceChildren(empty('カタログが空です。下のフォームで取得先を指定できます。')); return; }
  $('catalog').replaceChildren(...state.catalog.map(item => {
    const card = el('article', 'catalog-card'); card.append(el('h3', 'mono', item.repo_id));
    const localName = catalogLocalName(item), job = state.jobs.find(j => j.repo_id === item.repo_id);
    const local = state.models.find(m => m.id === (job?.local_name || localName));
    const candidate = local || state.models.find(m => m.id.split('/').at(-1) === localName.split('/').at(-1));
    const running = job && ['queued', 'downloading'].includes(job.status);
    const meta = el('div', 'catalog-meta'); meta.append(badge(item.type || '種別未判定'));
    meta.append(badge(running ? '取得中' : job?.status === 'failed' ? '取得失敗' : local ? 'ローカル検出' : job?.status === 'completed' ? '取得完了' : candidate ? '同名ローカルあり・出典未確認' : '未取得', running ? 'warn' : job?.status === 'failed' ? 'error' : ''));
    card.append(meta, el('div', 'muted mono', `${candidate && !job ? 'ローカル候補' : '保存先'}: ${job?.local_name || candidate?.id || localName}`), el('p', 'muted', job?.status === 'failed' ? '部分ファイルの可能性があります。ジョブのエラーを確認して再試行してください。' : candidate ? 'ロード・推論状態はモデル画面で確認できます。' : '取得後に互換性と推論を確認してください。'));
    const actions = el('div', 'card-actions');
    actions.append(action(running ? '取得中…' : job?.status === 'failed' ? '再試行' : candidate ? 'モデルを確認' : '取得を開始', () => candidate && !running && job?.status !== 'failed' ? handlers.models() : handlers.download(item.repo_id, job?.local_name || localName, job?.status === 'failed'), 'btn small', state.busy || running));
    card.append(actions); return card;
  }));
}

export function renderJobs(state, handlers) {
  const expanded = new Set([...$('jobs').querySelectorAll('details[open]')].map(n => n.closest('.job-row').dataset.id));
  if (!state.jobs.length) { $('jobs').replaceChildren(empty('ダウンロードジョブはありません。カタログからモデルを選択してください。')); return; }
  const labels = {queued: '待機中', downloading: '取得中', completed: '取得完了・推論未確認', failed: '取得失敗'};
  $('jobs').replaceChildren(...state.jobs.slice(0, 20).map(job => {
    const row = el('div', 'job-row'), heading = el('div', 'job-heading');
    row.dataset.id = job.id;
    heading.append(el('strong', 'mono', job.repo_id), badge(labels[job.status] || job.status, job.status === 'failed' ? 'error' : job.status === 'completed' ? '' : 'warn'));
    row.append(heading, el('div', 'muted mono', job.local_name));
    if (['queued', 'downloading'].includes(job.status)) {
      const progress = el('progress'); progress.max = 100; if (job.progress_percent != null) progress.value = Math.min(100, Math.max(0, job.progress_percent));
      progress.setAttribute('aria-label', `${job.repo_id} の取得進捗`); row.append(progress);
    }
    const percent = job.progress_percent == null ? '進捗率不明' : `${Number(job.progress_percent).toFixed(1)}%`;
    row.append(el('div', 'job-detail', `${percent} · ${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)} · ${job.speed_mbps ?? 0} MiB/s · ${formatTime(job.created_at)}`));
    if (job.error) {
      row.append(el('p', 'job-error', job.error), action('再試行', () => handlers.download(job.repo_id, job.local_name, true), 'btn small', state.busy));
    }
    if (job.last_log?.message) { const details = el('details'); details.open = expanded.has(job.id); details.append(el('summary', '', '最新の取得ログ'), el('pre', '', job.last_log.message)); row.append(details); }
    return row;
  }));
}

export function renderTestModels(state) {
  const select = $('test-model'), previous = select.value;
  const models = state.models.filter(m => m.loaded && m.type === 'embedding');
  // Keep selection and focus stable while the state is polled.
  const signature = JSON.stringify(models.map(m => m.id));
  if (select.dataset.models !== signature) {
    select.dataset.models = signature;
    select.replaceChildren(...models.map(m => { const option = el('option', '', m.id); option.value = m.id; return option; }));
    if (models.some(m => m.id === previous)) select.value = previous;
  }
  select.disabled = state.busy || !models.length; $('test-button').disabled = state.busy || !models.length;
  if (!models.length) $('test-validation').textContent = 'ロード済みの embedding モデルがありません。モデル画面でロードしてください。';
  else if ($('test-validation').textContent.startsWith('ロード済み')) $('test-validation').textContent = '';
}

export function renderTestResult(data, elapsed) {
  const ordered = [...(data.data || [])].sort((a, b) => a.index - b.index);
  const a = ordered[0]?.embedding, b = ordered[1]?.embedding;
  const similarity = cosineSimilarity(a, b), panel = el('div', 'panel');
  panel.append(el('h2', '', 'ベクトル生成が完了しました'), el('p', 'muted mono', data.model));
  const metrics = el('div', 'result-metrics');
  for (const [label, value] of [['Cosine similarity', similarity == null ? '算出不可（ゼロベクトル）' : similarity.toFixed(4)], ['出力次元', a.length], ['応答時間', `${elapsed.toFixed(0)} ms`]]) {
    const metric = el('div'); metric.append(el('div', 'muted', label), el('div', 'result-value', String(value))); metrics.append(metric);
  }
  panel.append(metrics, el('p', 'muted', `トークン数 ${data.usage?.total_tokens ?? '—'}（サーバーの概算）。類似度は品質の保証ではありません。`));
  const details = el('details'); details.append(el('summary', '', 'ベクトルの先頭 8 要素'), el('pre', '', JSON.stringify({A: a.slice(0, 8), B: b.slice(0, 8)}, null, 2)));
  panel.append(details); $('test-result').replaceChildren(panel);
}

export function renderActivity(state) {
  if (!state.logs.length) { $('activity').replaceChildren(empty('このプロセスの推論ログはまだありません。')); return; }
  const wrap = el('div', 'table-wrap'), table = el('table'), head = el('thead'), tr = el('tr');
  for (const title of ['日時 / 種別', 'モデル', '結果', '処理時間', '詳細']) tr.append(el('th', '', title));
  head.append(tr); table.append(head); const body = el('tbody');
  for (const log of [...state.logs].reverse()) {
    const row = el('tr'), first = el('td'); first.append(el('div', '', formatTime(log.timestamp)), el('span', 'muted', log.event));
    const status = el('td'); status.append(badge(log.details?.status === 'ok' ? '成功' : '失敗', log.details?.status === 'ok' ? 'good' : 'error'));
    const detail = el('td');
    if (log.details?.error) { const disclosure = el('details'); disclosure.append(el('summary', '', 'エラーを表示'), el('pre', '', log.details.error)); detail.append(disclosure); }
    else detail.textContent = `${log.details?.inputs_count ?? log.details?.documents_count ?? '—'} 件 · ${log.details?.dimensions_actual ?? '—'} 次元`;
    row.append(first, el('td', 'model-cell mono', log.model), status, el('td', '', `${log.duration_ms} ms`), detail); body.append(row);
  }
  table.append(body); wrap.append(table); $('activity').replaceChildren(wrap);
}

export function renderSettings(state) {
  const c = state.status?.config;
  if (!c) { $('settings').replaceChildren(empty('設定を取得できません。状態を更新してください。')); return; }
  const rows = [
    ['保存ディレクトリ', c.model_dir], ['デバイス設定', c.device_mode], ['リクエスト時の自動ロード', c.auto_load ? '有効' : '無効'],
    ['アイドル時の自動解放', c.idle_ttl ? `${c.idle_ttl} 秒` : '無効'], ['事前ロード · embedding', c.preload_embedding.join(', ') || 'なし'],
    ['事前ロード · reranker', c.preload_reranker.join(', ') || 'なし'], ['Chat proxy', c.chat_proxy_configured ? '設定済み' : '未設定'],
  ];
  $('settings').replaceChildren(...rows.flatMap(([label, value]) => [el('dt', '', label), el('dd', 'mono', String(value))]));
}
