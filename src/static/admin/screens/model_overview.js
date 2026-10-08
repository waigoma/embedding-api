/** Shared "モデル管理" screen: every model a service knows, grouped by family.
 *
 * Reads GET models/overview (feed key "overview"). Steps, actions, requests and
 * the fetch form all come from the server; this file never checks the service.
 * Services that need a custom navigation (e.g. tts opening a new voice) pass
 * hooks: createModelOverviewScreen({ hooks: { 'new-voice': (payload, ctx) => ... } }).
 */
import { defineScreen } from '../core/registry.js';
import { button, el, panelHeader } from '../core/dom.js';
import { formatBytes } from '../core/format.js';
import { json } from '../core/api.js';

const FEED = 'overview';
const ACTIVE_JOBS = new Set(['queued', 'downloading', 'cancelling']);
const FILTERS = [['all', 'すべて'], ['have', '取得済み'], ['rec', '推奨']];
const KIND_LABELS = { official: '公式', gateway: 'gateway', measured: '実測', unknown: '未確認' };

// ----- pure helpers (unit-tested) -----

/** Index of the furthest lifecycle step that is true, -1 when none. */
export function stageIndex(item, steps) {
  let reached = -1;
  steps.forEach((step, index) => { if (item.lifecycle?.[step.key]) reached = index; });
  return reached;
}

/** One-word state for the row and its tone (none | accent | ok | warn | muted). */
export function stageLabel(item, steps) {
  if (item.job && ACTIVE_JOBS.has(item.job.status)) return { text: item.job.status === 'cancelling' ? '取消中' : '取得中', tone: 'warn' };
  if (item.job && item.job.status === 'failed') return { text: '取得失敗', tone: 'bad' };
  if (item.support === 'candidate') return { text: '候補・未対応', tone: 'muted' };
  const index = stageIndex(item, steps);
  if (index < 0) return { text: '未取得', tone: 'none' };
  if (item.unverified || (item.support === 'local' && index === 0)) return { text: '取得済み・未検証', tone: 'accent' };
  const key = steps[index].key;
  return { text: steps[index].done_label, tone: key === 'loaded' || key === 'inference_confirmed' ? 'ok' : 'accent' };
}

/** The single action a row shows: first enabled primary, else first primary (disabled). */
export function primaryAction(item) {
  const primaries = (item.actions || []).filter(action => action.primary);
  return primaries.find(action => action.enabled) || primaries[0] || null;
}

export function matchesFilter(item, { filter = 'all', hideCandidates = false, query = '' } = {}) {
  if (filter === 'have' && !item.lifecycle?.downloaded) return false;
  if (filter === 'rec' && !item.recommended) return false;
  if (hideCandidates && item.support === 'candidate') return false;
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [item.title, item.id, item.family, item.version, item.purpose, item.group, item.api_name, item.local_name]
    .filter(Boolean).join(' ').toLowerCase().includes(q);
}

const SUPPORT_RANK = { adapter: 0, local: 1, candidate: 2 };
export function compareItems(a, b) {
  return (Number(Boolean(b.lifecycle?.downloaded)) - Number(Boolean(a.lifecycle?.downloaded)))
    || ((SUPPORT_RANK[a.support] ?? 0) - (SUPPORT_RANK[b.support] ?? 0))
    || (Number(Boolean(b.recommended)) - Number(Boolean(a.recommended)))
    || String(a.title).localeCompare(String(b.title), 'ja');
}

/** [{ family, items, total, downloaded, candidatesOnly, localOnly }] in display order. */
export function groupFamilies(allItems, shownItems) {
  const families = new Map();
  for (const item of [...shownItems].sort(compareItems)) {
    if (!families.has(item.family)) families.set(item.family, []);
    families.get(item.family).push(item);
  }
  const groups = [...families].map(([family, items]) => {
    const all = allItems.filter(item => item.family === family);
    return {
      family, items, total: all.length,
      downloaded: all.filter(item => item.lifecycle?.downloaded).length,
      candidatesOnly: all.every(item => item.support === 'candidate'),
      localOnly: all.every(item => item.support === 'local'),
    };
  });
  const score = g => g.downloaded * 100 - (g.candidatesOnly ? 1000 : 0) - (g.localOnly ? 500 : 0) + g.total;
  return groups.sort((a, b) => score(b) - score(a) || a.family.localeCompare(b.family, 'ja'));
}

export const OTHER_GROUP = 'その他';

/** Sub-group rows by item.group when a family is long and its rows carry groups.
 * Rows without a group (e.g. a stray カタログ外 directory) go to a trailing その他. */
export function subGroups(items, threshold = 8) {
  if (items.length <= threshold || !items.some(item => item.group)) return [{ group: null, items }];
  const groups = new Map();
  const rank = item => (item.group ? item.group_rank ?? 0 : Number.MAX_SAFE_INTEGER);
  const label = item => item.group || OTHER_GROUP;
  const ordered = [...items].sort((a, b) => (rank(a) - rank(b)) || label(a).localeCompare(label(b)) || compareItems(a, b));
  for (const item of ordered) {
    if (!groups.has(label(item))) groups.set(label(item), []);
    groups.get(label(item)).push(item);
  }
  return [...groups].map(([group, rows]) => ({ group, items: rows }));
}

/** Save name for the fetch form: the typed name, else prefix + repo name. */
export function resolveSaveName(repoId, typedName, prefix) {
  const typed = String(typedName || '').trim().replace(/^\/+|\/+$/g, '');
  if (typed) return typed;
  const base = String(repoId || '').trim().split('/').filter(Boolean).at(-1) || '';
  return base ? (prefix || '') + base : '';
}

export const isRepoId = value => /^[\w.-]+\/[\w.-]+$/.test(String(value || '').trim());

// ----- view -----

function rail(item, steps) {
  const node = el('span', 'mo-rail' + (item.support === 'candidate' ? ' cand' : '') + (item.lifecycle?.loaded ? ' loaded' : ''));
  node.setAttribute('aria-hidden', 'true');
  for (const step of steps) node.append(el('span', item.lifecycle?.[step.key] ? 'on' : ''));
  return node;
}

function sizeText(item) {
  const download = (item.specs || []).find(spec => spec.key === 'download');
  if (item.size_bytes != null) return { text: formatBytes(item.size_bytes), known: true };
  // The row has room for the size only; qualifiers ("· INT8") stay in the drawer.
  if (download && download.kind !== 'unknown') return { text: String(download.value).split(' · ')[0], known: true };
  return { text: '未確認', known: false };
}

function createState(hooks) {
  return { data: null, filter: 'all', hideCandidates: false, query: '', collapsed: new Set(), openId: null, mode: null, hooks, refs: {} };
}

export function createModelOverviewScreen({ id = 'models', section = 'Models', label = '▦ モデル管理', hooks = {} } = {}) {
  let state = null;
  let unsubscribe = null;
  let ctxRef = null;

  async function run(action, item, bodyPatch) {
    const ctx = ctxRef;
    if (action.navigate) {
      const hook = action.navigate.hook && state.hooks[action.navigate.hook];
      closeDrawer();
      if (hook) await hook(action.payload || {}, ctx, item);
      else await ctx.navigate({ id: action.navigate.tab, itemId: action.navigate.item ?? undefined });
      return;
    }
    if (!action.request) return;
    const { method = 'POST', path, body } = action.request;
    ctx.setBusy(true);
    try {
      const payload = body || bodyPatch ? { ...(body || {}), ...(bodyPatch || {}) } : undefined;
      await ctx.api(path, payload === undefined ? { method } : json(method, payload));
      await ctx.feed.refresh(FEED);
    } catch (error) {
      ctx.showError(error.message);
    } finally {
      ctx.setBusy(false);
    }
  }

  function trigger(action, item) {
    if (!action || !action.enabled) return;
    if (action.confirm) { openDrawer(item.id, action.id); return; }
    void run(action, item);
  }

  function actionButton(action, item, compact) {
    const tone = action.tone === 'danger' ? 'btn btn-danger'
      : action.tone === 'primary' || (compact && action.request) ? 'btn btn-primary' : 'btn btn-neutral';
    const label = compact ? action.short_label || action.label : action.label;
    const node = button(label, tone + (compact ? ' btn-sm' : ''), event => { event.stopPropagation(); trigger(action, item); });
    node.disabled = !action.enabled;
    if (!action.enabled && action.reason) node.title = action.reason;
    return node;
  }

  function rowAction(item) {
    const cell = el('div', 'mo-act');
    const job = item.job;
    if (job && ACTIVE_JOBS.has(job.status)) {
      const progress = el('div', 'mo-progress');
      const bar = el('i');
      const fill = el('b');
      if (job.progress_percent == null) bar.classList.add('indeterminate');
      else fill.style.width = Math.max(0, Math.min(100, job.progress_percent)) + '%';
      bar.append(fill);
      const text = job.progress_percent == null ? '計測中' : job.progress_percent.toFixed(0) + '%';
      progress.append(bar, el('small', '', text + (job.speed_mbps ? ' · ' + job.speed_mbps + ' MB/s' : '')));
      cell.append(progress);
      return cell;
    }
    if (item.support === 'candidate') { cell.append(el('span', 'mo-muted', '掲載のみ')); return cell; }
    const action = primaryAction(item);
    if (action) cell.append(actionButton(action, item, true));
    return cell;
  }

  function renderList() {
    const { data, refs } = state;
    if (!data || !refs.list) return;
    const steps = data.steps;
    const items = data.items || [];
    const count = key => items.filter(item => item.lifecycle?.[key]).length;
    refs.summary.replaceChildren();
    const sums = [['モデル', items.length], ['取得済み', count('downloaded')]];
    if (steps.some(step => step.key === 'loaded')) sums.push(['ロード中', count('loaded')]);
    if (steps.some(step => step.key === 'inference_confirmed')) sums.push(['推論確認', count('inference_confirmed')]);
    for (const [name, value] of sums) {
      const node = el('span');
      node.append(el('b', '', String(value)), ' ' + name);
      refs.summary.append(node);
    }
    refs.legend.replaceChildren(rail({ lifecycle: { [steps[0].key]: true, [steps[1]?.key]: true } }, steps), steps.map(step => step.label).join(' · '));
    refs.roots.textContent = '保存領域: ' + (data.roots || []).join(' / ');
    refs.addButton.hidden = !data.fetch;

    const shown = items.filter(item => matchesFilter(item, state));
    refs.list.replaceChildren();
    if (!shown.length) {
      refs.list.append(el('p', 'mo-empty', items.length ? '条件に合うモデルがありません。絞り込みを外してください。' : 'モデルがありません。'));
      return;
    }
    for (const group of groupFamilies(items, shown)) {
      const closed = state.collapsed.has(group.family) && !state.query;
      const sectionNode = el('section', 'mo-family' + (closed ? ' closed' : ''));
      const head = button('', 'mo-fam-head', () => {
        if (state.collapsed.has(group.family)) state.collapsed.delete(group.family); else state.collapsed.add(group.family);
        renderList();
      });
      head.setAttribute('aria-expanded', String(!closed));
      const meta = group.candidatesOnly ? `${group.total} モデル · 候補のみ`
        : `${group.downloaded} / ${group.total} 取得済み` + (group.items.length < group.total ? ` · ${group.items.length} 件表示` : '');
      const bar = el('span', 'mo-fam-bar');
      const fill = el('i');
      fill.style.width = (group.total ? group.downloaded / group.total * 100 : 0) + '%';
      bar.append(fill);
      head.append(el('span', 'mo-chev', '▾'), el('span', 'mo-fam-name', group.family), el('span', 'mo-fam-meta', meta), bar);
      const rows = el('div', 'mo-rows');
      const cols = el('div', 'mo-cols');
      for (const name of ['状態', 'モデル', data.version_label || '版', '取得サイズ', 'ライセンス', '', '']) cols.append(el('span', '', name));
      rows.append(cols);
      for (const sub of subGroups(group.items)) {
        if (sub.group) {
          const heading = el('div', 'mo-sub', sub.group);
          heading.append(el('span', '', String(sub.items.length)));
          rows.append(heading);
        }
        for (const item of sub.items) rows.append(renderRow(item, steps));
      }
      sectionNode.append(head, rows);
      refs.list.append(sectionNode);
    }
  }

  function renderRow(item, steps) {
    const row = el('div', 'mo-row' + (item.support === 'candidate' ? ' cand' : '') + (item.support === 'local' ? ' local' : ''));
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-label', item.title + ' の詳細を開く');
    if (item.id === state.openId) row.setAttribute('aria-current', 'true');
    const label = stageLabel(item, steps);
    const stateCell = el('div', 'mo-state');
    stateCell.append(rail(item, steps), el('small', 'tone-' + label.tone, label.text));
    stateCell.title = steps.map(step => (item.lifecycle?.[step.key] ? '✓ ' : '・ ') + step.label).join('\n');
    const name = el('div', 'mo-name');
    const title = el('b', '', item.title);
    if (item.recommended) title.append(el('span', 'mo-star', '★ 推奨'));
    if (item.support === 'local' && item.family !== 'カタログ外') title.append(el('span', 'mo-tag', 'カタログ外'));
    name.append(title, el('small', '', item.purpose || item.api_name || item.local_name || item.id));
    const size = sizeText(item);
    row.append(stateCell, name, el('span', 'mo-ver', item.version || ''), el('span', 'mo-size' + (size.known ? '' : ' unk'), size.text),
      el('span', 'mo-lic', item.license || '—'), rowAction(item), el('span', 'mo-go', '›'));
    const open = () => openDrawer(item.id);
    row.addEventListener('click', open);
    row.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
    return row;
  }

  // ----- drawer -----

  function ensureDrawer() {
    if (state.refs.drawer) return state.refs.drawer;
    const scrim = el('div', 'mo-scrim');
    scrim.hidden = true;
    scrim.addEventListener('click', closeDrawer);
    const drawer = el('aside', 'mo-drawer');
    drawer.hidden = true;
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-modal', 'true');
    document.body.append(scrim, drawer);
    state.refs.scrim = scrim;
    state.refs.drawer = drawer;
    return drawer;
  }

  function drawerHead(drawer, title, sub) {
    const head = el('div', 'mo-d-head');
    const heading = el('h2', '', title);
    heading.id = 'mo-d-title';
    drawer.setAttribute('aria-labelledby', heading.id);
    const close = button('×', 'mo-x', closeDrawer);
    close.setAttribute('aria-label', '閉じる');
    head.append(heading, close, el('code', '', sub));
    drawer.append(head);
    return close;
  }

  function openDrawer(itemId, confirmActionId = null) {
    const item = (state.data?.items || []).find(entry => entry.id === itemId);
    if (!item) return;
    state.openId = itemId;
    state.mode = 'item';
    const drawer = ensureDrawer();
    drawer.replaceChildren();
    const close = drawerHead(drawer, item.title, [item.id, item.family].filter(Boolean).join(' · '));
    const body = el('div', 'mo-d-body');
    if (item.purpose || item.note) {
      const intro = el('section', 'mo-intro');
      if (item.purpose) intro.append(el('p', '', item.purpose));
      if (item.note) intro.append(el('small', '', item.note));
      body.append(intro);
    }
    body.append(stepsSection(item), actionsSection(item, confirmActionId));
    if (item.api_name) body.append(apiSection(item));
    if ((item.specs || []).length) body.append(specsSection(item));
    body.append(storageSection(item));
    drawer.append(body);
    showDrawer();
    if (!confirmActionId) close.focus();
    renderList();
  }

  function stepsSection(item) {
    const node = el('section');
    node.append(el('h3', '', '状態'));
    const list = el('ol', 'mo-steps');
    for (const step of state.data.steps) {
      const done = item.lifecycle?.[step.key];
      const li = el('li', done ? 'done' + (step.key === 'loaded' || step.key === 'inference_confirmed' ? ' green' : '') : '');
      let note = done ? step.hint : '未確認';
      if (step.key === 'files_verified' && item.integrity) note = item.integrity;
      if (step.key === 'downloaded' && !done && item.job?.status === 'failed') note = '取得失敗: ' + (item.job.error || '');
      const text = el('div');
      text.append(el('b', '', step.label), el('small', '', note));
      li.append(el('span', 'mo-dot'), text);
      list.append(li);
    }
    if (item.job && ACTIVE_JOBS.has(item.job.status)) {
      const li = el('li', 'job');
      const text = el('div');
      text.append(el('b', '', '取得中'), el('small', '', item.job.progress_percent == null ? '進捗を計測中' : item.job.progress_percent.toFixed(1) + '%'));
      li.append(el('span', 'mo-dot'), text);
      list.prepend(li);
    }
    node.append(list);
    return node;
  }

  function actionsSection(item, confirmActionId) {
    const node = el('section');
    node.append(el('h3', '', '操作'));
    const list = el('div', 'mo-actions');
    const actions = [...(item.actions || [])];
    if (item.job?.cancel) actions.unshift(item.job.cancel);
    if (!actions.length) list.append(el('small', 'mo-muted', 'このモデルに実行できる操作はありません。'));
    for (const action of actions) {
      const row = el('div', 'mo-action');
      if (action.id === confirmActionId && action.confirm) {
        row.classList.add('confirm');
        row.append(el('b', '', action.confirm.title || action.label), el('p', '', action.confirm.text || ''));
        const buttons = el('div', 'btn-group');
        const yes = button(action.confirm.accept_label || action.label, action.tone === 'danger' ? 'btn btn-danger' : 'btn btn-primary', () => { void run(action, item, action.confirm.body_patch); openDrawer(item.id); });
        const no = button('やめる', 'btn btn-neutral', () => openDrawer(item.id));
        buttons.append(yes, no);
        row.append(buttons);
        list.append(row);
        queueMicrotask(() => yes.focus());
        continue;
      }
      const btn = button(action.label, action.tone === 'danger' ? 'btn btn-danger' : action.primary ? 'btn btn-primary' : 'btn btn-neutral', () => trigger(action, item));
      btn.disabled = !action.enabled;
      row.append(btn, el('small', '', action.reason || ''));
      list.append(row);
    }
    node.append(list);
    return node;
  }

  function apiSection(item) {
    const node = el('section');
    node.append(el('h3', '', 'API でのモデル名'));
    const row = el('div', 'mo-api');
    const copy = button('コピー', 'btn btn-neutral btn-sm', () => {
      navigator.clipboard?.writeText(item.api_name).then(() => { copy.textContent = 'コピーしました'; }, () => {});
    });
    row.append(el('code', '', '"model": "' + item.api_name + '"'), copy);
    node.append(row, el('small', 'mo-muted', '保存名がそのまま API の model になります。'));
    return node;
  }

  function specsSection(item) {
    const node = el('section');
    node.append(el('h3', '', '仕様と根拠'));
    const list = el('dl', 'mo-specs');
    for (const spec of item.specs) {
      const value = el('dd', '', spec.value || '—');
      if (spec.kind) value.append(el('span', 'mo-kind' + (spec.kind === 'official' ? ' official' : ''), KIND_LABELS[spec.kind] || spec.kind));
      if (spec.conditions) value.append(el('small', '', spec.conditions));
      if (spec.source) {
        const link = el('a', '', '出典');
        link.href = spec.source;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        const wrap = el('small');
        wrap.append(link);
        value.append(wrap);
      }
      list.append(el('dt', '', spec.label), value);
    }
    node.append(list);
    return node;
  }

  function storageSection(item) {
    const node = el('section');
    node.append(el('h3', '', '保存先と固定版'));
    for (const path of item.paths?.length ? item.paths : ['未取得 · 取得後に実パスを表示']) node.append(el('div', 'mo-path', path));
    const meta = el('div', 'mo-links');
    meta.append(el('span', '', '版: ' + (item.revision || '未固定')), el('span', '', 'ライセンス: ' + (item.license || '未確認')));
    if (item.source) {
      const link = el('a', '', '配布元を開く');
      link.href = item.source;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      meta.append(link);
    }
    node.append(meta);
    return node;
  }

  function openFetchForm() {
    const fetch = state.data?.fetch;
    if (!fetch) return;
    state.openId = null;
    state.mode = 'fetch';
    const drawer = ensureDrawer();
    drawer.replaceChildren();
    drawerHead(drawer, 'リポジトリから取得', 'Hugging Face のリポジトリを保存領域へ取得します');
    const body = el('div', 'mo-d-body');
    const form = el('form', 'mo-form');
    form.noValidate = true;
    const input = (idName, label, placeholder, help) => {
      const wrap = el('div', 'mo-field');
      const labelNode = el('label', '', label);
      labelNode.htmlFor = idName;
      const node = el('input');
      node.type = 'text';
      node.id = idName;
      node.placeholder = placeholder;
      node.autocomplete = 'off';
      node.spellcheck = false;
      wrap.append(labelNode, node);
      if (help) wrap.append(el('small', '', help));
      form.append(wrap);
      return node;
    };
    const repo = input('mo-repo', 'リポジトリ', 'owner/repo', '例: owner/model-name');
    const name = input('mo-name', '保存名', fetch.name_prefix + 'repo 名',
      fetch.api_name ? 'そのまま API の model 名になります。空ならリポジトリ名から作ります。' : '保存領域からの相対パスです。空ならリポジトリ名から作ります。');
    const extras = (fetch.fields || []).map(spec => [spec.key, input('mo-extra-' + spec.key, spec.label, spec.placeholder || '', spec.help)]);
    const preview = el('dl', 'mo-preview');
    const warn = el('div', 'mo-warnbox');
    warn.hidden = true;
    const submit = el('button', 'btn btn-primary', '取得を開始');
    submit.type = 'submit';
    form.append(preview, warn, submit);
    let edited = false;
    let force = null;
    const installed = new Set((state.data.items || []).filter(item => item.lifecycle?.downloaded && item.local_name).map(item => item.local_name));
    const update = () => {
      const saveName = resolveSaveName(repo.value, edited ? name.value : '', fetch.name_prefix);
      if (!edited) name.value = saveName;
      preview.replaceChildren(el('dt', '', '保存先'), el('dd', '', saveName ? (state.data.models_root || '') + saveName : '—'));
      if (fetch.api_name) preview.append(el('dt', '', 'API 名'), el('dd', '', saveName ? '"model": "' + saveName + '"' : '—'));
      warn.replaceChildren();
      warn.hidden = !installed.has(saveName);
      force = null;
      if (!warn.hidden) {
        warn.append(el('span', '', saveName + ' は取得済みです。上書きすると今のファイルを置き換えます。'));
        const label = el('label', 'field-label inline');
        force = el('input');
        force.type = 'checkbox';
        force.id = 'mo-force';
        label.append(force, el('span', '', '上書きして取得し直す'));
        warn.append(label);
      }
      submit.disabled = !isRepoId(repo.value) || !saveName;
    };
    repo.addEventListener('input', update);
    name.addEventListener('input', () => { edited = name.value.trim() !== ''; update(); });
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const saveName = resolveSaveName(repo.value, edited ? name.value : '', fetch.name_prefix);
      if (!isRepoId(repo.value) || !saveName) return;
      if (force && !force.checked) { ctxRef.showError('取得済みの保存名です。上書きする場合はチェックを入れてください'); return; }
      const extra = Object.fromEntries(extras.map(([key, node]) => [key, node.value.trim()]).filter(([, value]) => value !== ''));
      submit.disabled = true;
      ctxRef.setBusy(true);
      try {
        await ctxRef.api(fetch.request.path, json(fetch.request.method || 'POST', { repo_id: repo.value.trim(), local_name: saveName, force: Boolean(force?.checked), extra }));
        closeDrawer();
        await ctxRef.feed.refresh(FEED);
      } catch (error) {
        ctxRef.showError(error.message);
        submit.disabled = false;
      } finally {
        ctxRef.setBusy(false);
      }
    });
    body.append(form);
    drawer.append(body);
    update();
    showDrawer();
    repo.focus();
  }

  function showDrawer() {
    state.refs.drawer.hidden = false;
    state.refs.scrim.hidden = false;
  }

  function closeDrawer() {
    if (!state) return;
    state.openId = null;
    state.mode = null;
    if (state.refs.drawer) { state.refs.drawer.hidden = true; state.refs.scrim.hidden = true; }
    renderList();
  }

  const onKey = event => { if (event.key === 'Escape' && state?.mode) closeDrawer(); };

  function render(data) {
    if (!data || !state) return;
    state.data = data;
    renderList();
    // Keep an open detail panel current; never rebuild the fetch form under the user's typing.
    if (state.mode === 'item' && state.openId && !state.refs.drawer?.querySelector('.mo-action.confirm')) {
      if ((data.items || []).some(item => item.id === state.openId)) openDrawer(state.openId); else closeDrawer();
    }
  }

  return defineScreen({
    id, section, label,

    async mount(panel, ctx) {
      ctxRef = ctx;
      state = createState(hooks);
      const refresh = button('状態を更新', 'btn btn-neutral', () => ctx.feed.refresh(FEED).catch(error => ctx.showError(error.message)));
      const add = button('＋ リポジトリから取得', 'btn btn-primary', openFetchForm);
      add.hidden = true;
      panelHeader(panel, 'モデル管理', refresh, add);
      const summary = el('div', 'mo-summary');
      const toolbar = el('div', 'mo-toolbar');
      const search = el('input', 'field-input mo-search');
      search.type = 'search';
      search.id = 'mo-search';
      search.placeholder = 'モデル名・エンジン・言語で絞り込み';
      search.setAttribute('aria-label', 'モデルを検索');
      search.addEventListener('input', () => { state.query = search.value; renderList(); });
      const seg = el('div', 'mo-seg');
      seg.setAttribute('role', 'group');
      seg.setAttribute('aria-label', '表示');
      for (const [value, text] of FILTERS) {
        const node = button(text, '', () => {
          state.filter = value;
          for (const other of seg.children) other.setAttribute('aria-pressed', String(other === node));
          renderList();
        });
        node.setAttribute('aria-pressed', String(value === 'all'));
        seg.append(node);
      }
      const hide = el('label', 'field-label inline mo-check');
      const hideBox = el('input');
      hideBox.type = 'checkbox';
      hideBox.id = 'mo-hide-candidates';
      hideBox.addEventListener('change', () => { state.hideCandidates = hideBox.checked; renderList(); });
      hide.append(hideBox, el('span', '', '未対応の候補を隠す'));
      const legend = el('span', 'mo-legend');
      toolbar.append(search, seg, hide, legend);
      const roots = el('div', 'mo-roots');
      const list = el('div', 'mo-list');
      panel.append(summary, toolbar, roots, list);
      state.refs = { summary, legend, roots, list, addButton: add };
      document.addEventListener('keydown', onKey);
      render(ctx.feed.cached(FEED) || await ctx.api('models/overview'));
      unsubscribe = ctx.feed.subscribe(FEED, render);
    },

    unmount() {
      if (unsubscribe) unsubscribe();
      unsubscribe = null;
      document.removeEventListener('keydown', onKey);
      state?.refs.drawer?.remove();
      state?.refs.scrim?.remove();
      state = null;
    },
  });
}

export default createModelOverviewScreen();
