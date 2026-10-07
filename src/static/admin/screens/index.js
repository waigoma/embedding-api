/** Screen registry: the ONLY place a service edits to add, remove or reorder
 * sidebar entries. Order here is sidebar order; consecutive equal sections share
 * one label. feedLoaders adds service-specific snapshot keys (key -> api => promise).
 */
import { defineScreen } from '../core/registry.js';
import models from './models.js';
import catalog from './catalog.js';
import playground from './playground.js';
import recent from './recent.js';
import health from './health.js';

// 'health' pairs with the hub provider of the same key (/admin/health).
export const feedLoaders = { health: api => api('health') };
// Old `?tab=` names that should open a screen under a new id, and the screen for unknown tabs.
// activity / settings are the previous /ui page names.
export const tabAliases = { activity: 'recent', settings: 'health' };
export const fallbackTab = 'models';

export default [
  models,
  catalog,
  playground,
  defineScreen({ ...recent, label: '📊 Inference Logs' }),
  health,
];
