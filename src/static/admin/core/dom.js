/** Small DOM builders. Every service string goes through textContent; no HTML string assignment. */
export function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== '') node.textContent = text;
  return node;
}

export function button(text, className, onClick) {
  const node = el('button', className, text);
  node.type = 'button';
  if (onClick) node.addEventListener('click', onClick);
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

/** Standard labelled field inside a .field-block. Returns the input element. */
export function field(parent, { label, id, tag = 'input', type, value = '', placeholder, help, options, className = 'field-block' }) {
  const block = el('div', className);
  const labelNode = el('label', 'field-label', label);
  labelNode.htmlFor = id;
  const classes = { textarea: 'field-textarea', select: 'field-select', input: 'field-input' };
  const input = el(tag, classes[tag] || 'field-input');
  input.id = id;
  if (type) input.type = type;
  if (placeholder) input.placeholder = placeholder;
  if (tag === 'select') {
    for (const option of options || []) {
      const node = el('option', '', option);
      node.value = option;
      input.append(node);
    }
  }
  input.value = value == null ? '' : String(value);
  block.append(labelNode, input);
  if (help) block.append(el('div', 'field-help', help));
  parent.append(block);
  return input;
}

export function checkbox(parent, { label, id, checked = false, className = 'field-block' }) {
  const block = el('div', className);
  const labelNode = el('label', 'field-label inline');
  const input = el('input');
  input.type = 'checkbox';
  input.id = id;
  input.checked = checked;
  labelNode.append(input, el('span', '', label));
  block.append(labelNode);
  parent.append(block);
  return input;
}

export function panelHeader(panel, title, ...actions) {
  const header = el('div', 'panel-header');
  header.append(el('h1', 'panel-title', title));
  if (actions.length) {
    const group = el('div', 'panel-actions');
    group.append(...actions);
    header.append(group);
  }
  panel.append(header);
  return header;
}

export function placeholder(text) {
  return el('div', 'placeholder', text);
}

export function keyValueRow(parent, key, value, className = '') {
  const row = el('div', 'kv-row');
  row.append(el('span', 'kv-key', key), el('span', 'kv-val' + (className ? ' ' + className : ''), value));
  parent.append(row);
  return row;
}
