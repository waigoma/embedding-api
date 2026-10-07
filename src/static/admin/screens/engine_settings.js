/** Shared Engine Settings: forms rendered from server field specs. */
import { defineScreen } from '../core/registry.js';
import { button, el, panelHeader } from '../core/dom.js';
import { json } from '../core/api.js';

const FIELD_BUILDERS = Object.freeze({
  select(spec) {
    const input = el('select', 'field-select');
    for (const option of spec.options || []) {
      const node = el('option', '', String(option));
      node.value = String(option);
      if (String(option) === String(spec.value)) node.selected = true;
      input.append(node);
    }
    return { input, read: () => input.value };
  },
  text(spec) {
    const input = el('input', 'field-input');
    input.type = 'text';
    input.value = spec.value == null ? '' : String(spec.value);
    return { input, read: () => input.value.trim() };
  },
  number(spec) {
    const input = el('input', 'field-input');
    input.type = 'number';
    input.value = spec.value == null ? '' : String(spec.value);
    return { input, read: () => (input.value.trim() === '' ? null : Number(input.value)) };
  },
  checkbox(spec) {
    const input = el('input');
    input.type = 'checkbox';
    input.checked = Boolean(spec.value);
    return { input, read: () => input.checked };
  },
});

export function buildForm(container, form, ctx) {
  const card = el('div', 'card');
  card.append(el('div', 'card-title', form.title || form.name));
  if (form.note) card.append(el('div', 'field-help', form.note));
  const body = el('div', 'eng-form');
  const readers = {};
  for (const spec of form.fields) {
    const builder = FIELD_BUILDERS[spec.kind];
    if (!builder) throw new Error('Unknown field kind: ' + spec.kind);
    const wrap = el('div', 'eng-field');
    const { input, read } = builder(spec);
    input.id = 'eng-' + form.name + '-' + spec.key;
    const label = el('label', spec.kind === 'checkbox' ? 'field-label inline' : '', spec.label);
    label.htmlFor = input.id;
    if (spec.kind === 'checkbox') { label.prepend(input); wrap.append(label); }
    else wrap.append(label, input);
    if (spec.help) wrap.append(el('div', 'field-help', spec.help));
    readers[spec.key] = read;
    body.append(wrap);
  }
  const result = el('div', 'form-result');
  const apply = button('Apply', 'btn btn-primary', async () => {
    apply.disabled = true;
    result.className = 'form-result';
    result.textContent = '送信中…';
    const values = Object.fromEntries(Object.entries(readers).map(([key, read]) => [key, read()]));
    try {
      const applied = await ctx.api('engine-settings/' + encodeURIComponent(form.name), json('POST', values));
      result.textContent = '✓ Applied (' + Object.entries(applied).map(([key, value]) => key + '=' + String(value)).join(', ') + ')';
    } catch (error) {
      result.className = 'form-result err';
      result.textContent = '✗ ' + error.message;
    } finally {
      apply.disabled = false;
    }
  });
  apply.style.alignSelf = 'flex-start';
  body.append(apply, result);
  card.append(body);
  container.append(card);
}

export default defineScreen({
  id: 'engine-settings',
  section: 'Runtime',
  label: '⚙ Engine Settings',

  async mount(panel, ctx) {
    panelHeader(panel, 'Engine Settings');
    const data = await ctx.api('engine-settings');
    const forms = Array.isArray(data?.engines) ? data.engines : [];
    if (!forms.length) {
      panel.append(el('div', 'placeholder', '設定可能なエンジンがありません'));
      return;
    }
    for (const form of forms) buildForm(panel, form, ctx);
  },

  unmount() {},
});
