// Real Chromium against tests/serve_ui_fixture.py (real FastAPI + SSE hub, stubbed models/downloads).
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const origin = 'http://127.0.0.1:19975';
const output = process.env.UI_ARTIFACT_DIR || path.resolve('artifacts');
const SIDEBAR = ['↓ Model Downloads', '☆ カタログ', '▷ Playground', '📊 Inference Logs', '♥ Server Health'];

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH, headless: true, args: ['--no-sandbox']});
  const checks = [];
  const check = async (name, action) => { await action(); checks.push(name); console.log('PASS', name); };
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    const posts = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('request', request => { if (request.method() === 'POST') posts.push({url: new URL(request.url()).pathname, body: request.postData()}); });
    await page.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    const shot = name => page.screenshot({path: path.join(output, `embedding-${name}.png`), fullPage: true});
    const row = (table, text) => page.locator(`${table} tbody tr`, {hasText: text});

    await check('legacy /ui redirects to the admin shell; sidebar comes from the registry', async () => {
      await page.goto(origin + '/ui');
      assert.equal(new URL(page.url()).pathname, '/admin/ui');
      await page.waitForSelector('#sidebar-nav .sidebar-item');
      assert.deepEqual(await page.locator('#sidebar-nav .sidebar-item').allTextContents(), SIDEBAR);
      await page.locator('#feed-status', {hasText: 'SSE'}).waitFor();
      assert.equal(posts.length, 0, 'opening the UI performs no mutation');
    });

    await check('models: server columns, literal hostile name, load gated by a failed download', async () => {
      await page.getByRole('button', {name: SIDEBAR[0]}).click();
      await row('#dl-table', 'Qwen3-Embedding-0.6B').waitFor();
      assert.deepEqual(await page.locator('#dl-table th').allTextContents(),
        ['status', 'local_name', 'loaded', 'repo_id', 'progress', 'speed', 'size', 'error', 'type', 'actions']);
      assert.equal(await page.locator('#panel img').count(), 0, 'model names are rendered as text');
      await row('#dl-table', '<img onerror=alert(1)>').waitFor();
      assert.match(await row('#dl-table', 'reranker/synthetic').first().innerText(), /reranker/);
      await row('#dl-table', 'synthetic network timeout').waitFor();
      await shot('models-desktop');
      const partial = row('#dl-table', 'reranker/synthetic').filter({hasText: 'installed'});
      await partial.getByRole('button', {name: 'Load', exact: true}).click();
      await page.locator('#err-banner', {hasText: 'files may be partial'}).waitFor();
    });

    await check('models: Load / Unload round trip updates the table', async () => {
      const qwen = row('#dl-table', 'Qwen3-Embedding-0.6B');
      await qwen.getByRole('button', {name: 'Load', exact: true}).click();
      await qwen.getByRole('button', {name: 'Unload', exact: true}).waitFor();
      assert.ok(posts.some(p => p.url === '/admin/models/Qwen3-Embedding-0.6B/load'));
      await qwen.getByRole('button', {name: 'Unload', exact: true}).click();
      await qwen.getByRole('button', {name: 'Load', exact: true}).waitFor();
      await qwen.getByRole('button', {name: 'Load', exact: true}).click();
      await qwen.getByRole('button', {name: 'Unload', exact: true}).waitFor();
    });

    await check('catalog: preset entries, one-click download reaches installed via the feed', async () => {
      await page.getByRole('button', {name: SIDEBAR[1]}).click();
      await row('#catalog-table', 'Qwen/Qwen3-Embedding-4B').waitFor();
      assert.equal(await page.locator('#catalog-table tbody tr').count(), 6);
      assert.match(await row('#catalog-table', 'Qwen/Qwen3-Embedding-0.6B').innerText(), /installed/);
      await row('#catalog-table', 'Qwen/Qwen3-Embedding-4B').getByRole('button', {name: '取得', exact: true}).click();
      await row('#catalog-table', 'Qwen/Qwen3-Embedding-4B').filter({hasText: 'installed'}).waitFor({timeout: 15000});
      const download = posts.find(p => p.url === '/admin/models/download');
      assert.deepEqual(JSON.parse(download.body), {repo_id: 'Qwen/Qwen3-Embedding-4B', local_name: 'Qwen3-Embedding-4B', force: false});
      await shot('catalog-desktop');
    });

    await check('playground: dimension error is shown, then vectors and cosine similarity', async () => {
      await page.getByRole('button', {name: SIDEBAR[2]}).click();
      // The hub ticks once a second, so the just-loaded model may arrive one snapshot later.
      await page.locator('#pg-model option', {hasText: 'Qwen3-Embedding-0.6B'}).waitFor({state: 'attached'});
      assert.deepEqual(await page.locator('#pg-model option').allTextContents(), ['Qwen3-Embedding-0.6B', 'embedding/ruri-v3-310m']);
      await page.locator('#pg-dimensions').fill('9');
      await page.locator('#pg-run').click();
      await page.locator('#pg-validation', {hasText: '400'}).waitFor();
      await page.locator('#pg-dimensions').fill('4');
      await page.locator('#pg-run').click();
      await page.getByText('ベクトル生成が完了しました', {exact: true}).waitFor();
      assert.match(await page.locator('#pg-result').innerText(), /0\.9\d{3}/);
      assert.ok(posts.some(p => p.url === '/v1/embeddings'));
      await shot('playground-desktop');
    });

    await check('inference logs and server health read the feed', async () => {
      await page.getByRole('button', {name: SIDEBAR[3]}).click();
      await row('#obs-table', 'embedding').first().waitFor();
      assert.deepEqual(await page.locator('#obs-table th').allTextContents(), ['Time', 'Event', 'Model', 'Status', 'Duration']);
      await page.locator('#obs-table .status-fail').first().waitFor();
      await page.getByRole('button', {name: SIDEBAR[4]}).click();
      await page.locator('#health-body', {hasText: 'Synthetic GPU'}).waitFor();
      assert.match(await page.locator('#health-body').innerText(), /推論確認/);
      assert.match(await page.locator('#health-body').innerText(), /Chat proxy/);
      await shot('health-desktop');
    });

    await check('mobile layout has no horizontal overflow', async () => {
      await page.setViewportSize({width: 390, height: 844});
      for (const id of ['models', 'catalog', 'playground']) {
        await page.goto(`${origin}/admin/ui?mobile=${id}#${id}`);
        await page.locator('#panel .panel-title').waitFor();
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id);
      }
      await shot('playground-mobile');
    });

    await check('a reverse-proxy prefix keeps admin and /v1 requests under it', async () => {
      await page.setViewportSize({width: 1440, height: 1000});
      await page.goto(origin + '/prefix/webui');
      assert.equal(new URL(page.url()).pathname, '/prefix/admin/ui');
      await page.goto(origin + '/prefix/admin/ui?p=1#catalog');
      await row('#catalog-table', 'Qwen/Qwen3-Embedding-4B').waitFor();
    });

    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-qa.json'), JSON.stringify({fixture: true, production_inference: false, checks, page_errors: errors}, null, 2));
    console.log('Browser QA passed:', checks.length, 'scenarios');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
