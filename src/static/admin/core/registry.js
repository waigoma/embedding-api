/** Screen registry: the one place that lists what the sidebar shows.
 * A screen is { id, section, label, mount(panel, ctx), unmount() } plus optional
 * items(ctx) / mountItem(panel, ctx, itemId) for sidebar-listed entities,
 * sidebar(ctx) -> { before: Node[], after: Node[] } for extra sidebar controls,
 * init(ctx) run once at boot (e.g. to load the list behind items()), and
 * knownItem(id, ctx) when deep links may name items that items() does not list.
 */
const REQUIRED = ['id', 'section', 'label', 'mount', 'unmount'];
const OPTIONAL_FUNCTIONS = ['items', 'mountItem', 'sidebar', 'init', 'knownItem'];

export function defineScreen(definition) {
  for (const key of REQUIRED) {
    if (!(key in definition)) throw new Error('Screen definition missing ' + key);
  }
  for (const key of ['mount', 'unmount', ...OPTIONAL_FUNCTIONS]) {
    if (key in definition && typeof definition[key] !== 'function') throw new Error('Screen ' + definition.id + ': ' + key + ' must be a function');
  }
  if (typeof definition.id !== 'string' || !/^[a-z][a-z0-9_-]*$/.test(definition.id)) throw new Error('Screen id must be a lowercase slug: ' + definition.id);
  if (('items' in definition) !== ('mountItem' in definition)) throw new Error('Screen ' + definition.id + ': items and mountItem go together');
  return Object.freeze({ ...definition });
}

export function validateRegistry(screens) {
  if (!Array.isArray(screens) || !screens.length) throw new Error('Screen registry must be a non-empty array');
  const seen = new Set();
  for (const screen of screens) {
    for (const key of REQUIRED) {
      if (!(key in screen)) throw new Error('Registry entry is not a defined screen (missing ' + key + ')');
    }
    if (seen.has(screen.id)) throw new Error('Duplicate screen id: ' + screen.id);
    seen.add(screen.id);
  }
  return screens;
}

/** Group consecutive screens by section, preserving registry order. */
export function groupBySection(screens) {
  const groups = [];
  for (const screen of screens) {
    const last = groups[groups.length - 1];
    if (last && last.section === screen.section) last.screens.push(screen);
    else groups.push({ section: screen.section, screens: [screen] });
  }
  return groups;
}
