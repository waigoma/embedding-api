/** Screen registry: the ONLY place a service edits to add, remove or reorder
 * sidebar entries. Order here is sidebar order; consecutive equal sections share
 * one label. feedLoaders adds service-specific snapshot keys (key -> api => promise).
 */
import { defineScreen } from '../core/registry.js';
import { createModelOverviewScreen } from './model_overview.js';
import playground from './playground.js';
import recent from './recent.js';
import health from './health.js';

// 'health' pairs with the hub provider of the same key (/admin/health).
export const feedLoaders = { health: api => api('health'), overview: api => api('models/overview') };
// Old `?tab=` names that should open a screen under a new id, and the screen for unknown tabs.
// activity / settings are the previous /ui page names.
export const tabAliases = { activity: 'recent', settings: 'health', catalog: 'models', downloads: 'models' };
export const fallbackTab = 'models';

export default [
  createModelOverviewScreen({ id: 'models', section: 'Models', label: '▦ モデル管理' }),
  playground,
  defineScreen({ ...recent, label: '📊 Inference Logs' }),
  health,
];
