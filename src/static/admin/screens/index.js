/** embedding-api screen registry: sidebar order and sections. */
import models from './models.js';
import recent from './recent.js';
import health from './health.js';

// Extra snapshot key: the shell resolves 'health' through ctx.api (admin base relative).
export const feedLoaders = { health: api => api('health') };

export default [
  models,
  { ...recent, section: 'Observability', label: '📊 Inference Logs' },
  health,
];
