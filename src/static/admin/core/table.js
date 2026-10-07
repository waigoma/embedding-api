/** Column-spec driven table renderer. Server declares {key,label,kind}; this file
 * owns one renderer per kind. Unknown kinds throw so a contract drift is loud.
 */
import { button, clear, el } from './dom.js';
import { formatBytes, formatDuration, formatRelativeTime } from './format.js';
import { presentStatus } from './status.js';

function textCell(value, className = '') {
  return el('td', className, value === null || value === undefined ? '' : String(value));
}

export async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}

function copyButton(text) {
  const node = button('📋', 'btn-copy', async () => {
    try {
      await copyText(text);
      node.textContent = '✓';
    } catch {
      node.textContent = '✗';
    }
    setTimeout(() => { node.textContent = '📋'; }, 1200);
  });
  node.title = 'パスをコピー';
  return node;
}

export const CELL_RENDERERS = Object.freeze({
  text: value => textCell(value),
  mono: value => textCell(value, 'cell-mono'),
  bytes: value => textCell(formatBytes(value), 'cell-mono'),
  duration: value => textCell(formatDuration(value), 'cell-mono'),
  status: value => {
    const { className, label } = presentStatus(value);
    return el('td', className, label);
  },
  time: (value, options) => {
    const cell = el('td', '', formatRelativeTime(value, options.now));
    if (value) cell.title = String(value);
    return cell;
  },
  'path-list': (value, options) => {
    const cell = el('td', 'cell-paths');
    const paths = Array.isArray(value) ? value : [];
    if (!paths.length) {
      cell.textContent = '-';
      cell.classList.add('cell-mono');
      return cell;
    }
    for (const relative of paths) {
      const full = options.pathPrefix ? options.pathPrefix.replace(/\/$/, '') + '/' + relative : String(relative);
      const row = el('div', 'cell-path');
      row.append(el('span', '', full), copyButton(full));
      cell.append(row);
    }
    return cell;
  },
  actions: (_value, options, item) => {
    const cell = el('td', 'cell-actions');
    for (const action of options.actions ? options.actions(item) : []) {
      cell.append(button(action.label, 'btn btn-sm ' + (action.className || 'btn-outline'), action.onClick));
    }
    return cell;
  },
});

export function cellValue(item, key) {
  if (item && Object.prototype.hasOwnProperty.call(item, key)) return item[key];
  if (item && item.extra && Object.prototype.hasOwnProperty.call(item.extra, key)) return item.extra[key];
  return undefined;
}

/**
 * @param {HTMLTableElement} table
 * @param {{key:string,label:string,kind:string}[]} columns
 * @param {object[]} items
 * @param {{actions?:(item)=>{label,className,onClick}[], emptyText?:string, pathPrefix?:string, now?:number, errorKey?:string}} options
 */
export function renderTable(table, columns, items, options = {}) {
  for (const column of columns) {
    if (!CELL_RENDERERS[column.kind]) throw new Error('Unknown cell kind: ' + column.kind + ' (column ' + column.key + ')');
  }
  clear(table);
  const head = el('thead');
  const headings = el('tr');
  for (const column of columns) {
    const cell = el('th', '', column.label);
    cell.scope = 'col';
    headings.append(cell);
  }
  head.append(headings);
  const body = el('tbody');
  if (!items.length) {
    const row = el('tr');
    const cell = el('td', 'obs-empty', options.emptyText || 'まだデータがありません');
    cell.colSpan = Math.max(1, columns.length);
    row.append(cell);
    body.append(row);
  }
  for (const item of items) {
    const row = el('tr');
    for (const column of columns) {
      const cell = CELL_RENDERERS[column.kind](cellValue(item, column.key), options, item);
      if (options.errorKey && column.key === options.errorKey) cell.classList.add('cell-error');
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
}
