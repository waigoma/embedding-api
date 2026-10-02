import {request, snapshot} from './api.js';
import {pages, validateDownload, testPayload} from './state.js';
import {$, renderRuntime, renderModels, renderCatalog, renderJobs, renderTestModels, renderTestResult, renderActivity, renderSettings} from './view.js';

const state = {page: 'models', health: null, models: [], catalog: [], jobs: [], logs: [], status: null, busy: false, failures: []};
let timer, controller, refreshTask, noticeTimer, confirming = false;
const forms = ['download-form', 'test-form'];
let dirty = false;
let operationError = '', readError = '';

function message(text, error = false, source = 'operation') {
  if (error) {
    if (source === 'read') readError = text; else operationError = text;
    text = [operationError, readError].filter(Boolean).join(' / ');
  }
  const target = $(error ? 'error' : 'notice'); target.textContent = text; target.hidden = !text;
  if (!error) { clearTimeout(noticeTimer); if (text) noticeTimer = setTimeout(() => { target.hidden = true; }, 8000); }
}

function page(name, focus = true) {
  if (!pages[name] || state.busy) return;
  state.page = name;
  document.querySelectorAll('.page').forEach(node => { node.hidden = node.id !== `page-${name}`; });
  document.querySelectorAll('[data-page]').forEach(node => {
    const active = node.dataset.page === name; node.classList.toggle('active', active);
    if (active) node.setAttribute('aria-current', 'page'); else node.removeAttribute('aria-current');
  });
  [$('page-title').textContent, $('page-description').textContent] = pages[name];
  if (focus) $('main').focus({preventScroll: true});
}

function render() {
  // Keep actions disabled when the latest snapshot is incomplete.
  const viewState = {...state, busy: state.busy || state.failures.length > 0};
  renderRuntime(state); renderModels(viewState, handlers); renderCatalog(viewState, handlers);
  renderJobs(viewState, handlers); renderTestModels(viewState); renderActivity(state); renderSettings(state);
  document.querySelectorAll('[data-page]').forEach(n => { n.disabled = state.busy; });
  for (const id of forms) $(id).querySelectorAll('input, textarea, button').forEach(n => { n.disabled = viewState.busy; });
  $('test-button').disabled = viewState.busy || !$('test-model').options.length;
}

async function refresh() {
  if (refreshTask) return refreshTask;
  if (state.busy) return;
  clearTimeout(timer);
  controller = new AbortController();
  $('refresh').disabled = true;
  refreshTask = (async () => {
    try {
      const {values, failures} = await snapshot(controller.signal);
      for (const [key, value] of Object.entries(values)) state[key] = ['models', 'catalog', 'jobs', 'logs'].includes(key) ? value.data : value;
      state.failures = failures;
      const connected = !failures.length;
      $('connection').textContent = connected ? '接続済み' : '取得失敗・操作停止';
      $('connection').className = `badge ${connected ? 'good' : 'error'}`;
      $('updated').textContent = `${connected ? '最終更新' : '一部の状態が未取得'} ${new Date().toLocaleTimeString('ja-JP')}`;
      if (failures.length) message(`状態の取得に失敗しました。接続を確認し「状態を更新」を押してください。 ${failures.join(' / ')}`, true, 'read');
      else message('', true, 'read');
      $('initial').hidden = true; page(state.page, false); render();
    } catch (error) {
      if (error.name !== 'AbortError') message(error.message, true, 'read');
    } finally {
      refreshTask = null; $('refresh').disabled = false;
      if (!document.hidden && !state.busy) timer = setTimeout(refresh, 5000);
    }
  })();
  return refreshTask;
}

function confirm(title, description, label) {
  if (confirming) return Promise.resolve(false);
  confirming = true;
  const dialog = $('confirm-dialog'); $('confirm-title').textContent = title;
  $('confirm-message').textContent = description; $('confirm-action').textContent = label;
  dialog.returnValue = ''; dialog.showModal();
  // The cancel button receives focus, so Enter cannot unexpectedly confirm.
  dialog.querySelector('[value="cancel"]').focus();
  return new Promise(resolve => dialog.addEventListener('close', () => {
    confirming = false; resolve(dialog.returnValue === 'confirm');
  }, {once: true}));
}

async function mutate(path, body, success) {
  if (state.busy || state.failures.length) return;
  state.busy = true; render(); message('', true); clearTimeout(timer);
  // Let the previous read finish before the write and its follow-up snapshot.
  if (refreshTask) await refreshTask;
  if (state.failures.length) { state.busy = false; render(); return; }
  let wrote = false;
  let errorMessage = '';
  try {
    await request(path, {body}); wrote = true; message(success);
  } catch (error) { errorMessage = error.message; }
  finally {
    state.busy = false;
    await refresh();
    if (wrote && state.failures.length) message(`${success} 状態の再取得に失敗しました。「状態を更新」で確認してください。`);
    if (errorMessage) message(errorMessage, true);
    render();
  }
}

const handlers = {
  models: () => page('models'),
  test: id => { page('playground'); $('test-model').value = id; $('test-result').replaceChildren(); },
  load: async (id, type) => {
    if (await confirm('モデルをロード', `${id} を ${type} としてロードします。メモリを使用し、他の処理と競合する可能性があります。`, 'ロード'))
      await mutate('v1/models/load', {model_id: id, model_type: type}, 'ロードが完了しました。推論はまだ確認していません。');
  },
  unload: async id => {
    if (await confirm('モデルをアンロード', `${id} を解放します。このモデルを利用するクライアントの処理に影響する可能性があります。`, 'アンロード'))
      await mutate('v1/models/unload', {model_id: id}, 'モデルをアンロードしました。');
  },
  download: async (repoId, localName, force = false) => {
    const error = validateDownload(repoId, localName);
    if (error) { message(error, true); return; }
    if (await confirm(force ? 'モデルを再取得' : 'モデルを取得', `${repoId} を ${localName || repoId.split('/').at(-1)} に取得します。ネットワークとディスクを使用します。${force ? '既存の部分ファイルを使って再取得します。' : ''}ロードは行いません。`, force ? '再取得' : '取得を開始'))
      await mutate('v1/models/download', {repo_id: repoId, local_name: localName || null, force}, 'ダウンロードを開始しました。ジョブ一覧で進捗を確認できます。');
  },
};

$('download-form').addEventListener('submit', async event => {
  event.preventDefault();
  const repoId = $('repo-id').value.trim(), localName = $('local-name').value.trim();
  const error = validateDownload(repoId, localName); $('download-validation').textContent = error;
  if (!error) await handlers.download(repoId, localName, $('force').checked);
});

$('test-form').addEventListener('submit', async event => {
  event.preventDefault(); if (state.busy || state.failures.length) return;
  $('test-validation').textContent = ''; $('test-result').replaceChildren();
  let body;
  try { body = testPayload($('test-model').value, $('text-a').value, $('text-b').value, $('dimensions').value); }
  catch (error) { $('test-validation').textContent = error.message; return; }
  if (!state.models.some(m => m.id === body.model && m.loaded && m.type === 'embedding')) return;
  state.busy = true; render(); clearTimeout(timer); $('test-button').textContent = '生成中…';
  if (refreshTask) await refreshTask;
  if (state.failures.length || !state.models.some(m => m.id === body.model && m.loaded && m.type === 'embedding')) {
    state.busy = false; $('test-button').textContent = 'ベクトルを生成'; render();
    $('test-validation').textContent = '最新のロード状態を確認できません。状態を更新してから実行してください。';
    return;
  }
  const started = performance.now();
  let errorMessage = '';
  try { renderTestResult(await request('v1/embeddings', {body}), performance.now() - started); }
  catch (error) { errorMessage = error.message; }
  finally {
    state.busy = false; $('test-button').textContent = 'ベクトルを生成';
    await refresh(); render(); if (errorMessage) $('test-validation').textContent = errorMessage;
  }
});

$('model-search').addEventListener('input', () => renderModels({...state, busy: state.busy || state.failures.length > 0}, handlers));
$('test-model').addEventListener('change', () => $('test-result').replaceChildren());
$('refresh').addEventListener('click', refresh);
for (const node of document.querySelectorAll('[data-page]')) node.addEventListener('click', () => page(node.dataset.page));
for (const id of forms) $(id).addEventListener('input', () => { dirty = true; });
window.addEventListener('beforeunload', event => { if (dirty || state.busy) { event.preventDefault(); event.returnValue = ''; } });
document.addEventListener('visibilitychange', () => {
  clearTimeout(timer);
  if (document.hidden) { if (!state.busy) controller?.abort(); }
  else refresh();
});
window.addEventListener('pagehide', () => { clearTimeout(timer); controller?.abort(); });
refresh();
