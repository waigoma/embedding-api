// Real Chromium against tests/serve_ui_fixture.py (real FastAPI + SSE hub, stubbed models/downloads).
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const origin = 'http://127.0.0.1:19975';
const output = process.env.UI_ARTIFACT_DIR || path.resolve('artifacts');
const SIDEBAR = ['▦ モデル管理', '▷ Playground', '📊 Inference Logs', '♥ Server Health'];

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
    page.on('console', message => { if (message.type() === 'error' && !/status of 400/.test(message.text())) errors.push(message.text()); });  // 400: the playground's deliberate dimension error
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

    const family = name => page.locator('.mo-family', {has: page.locator('.mo-fam-name', {hasText: name})});
    const modelRow = title => page.locator('.mo-row', {hasText: title});

    await check('models: families, roots scope, literal hostile name', async () => {
      await page.getByRole('button', {name: SIDEBAR[0]}).click();
      await modelRow('Qwen3-Embedding-4B').waitFor();
      for (const name of ['Qwen3-Embedding', 'Qwen3-Reranker', 'Ruri', 'カタログ外']) await family(name).waitFor();
      assert.equal(await page.locator('.mo-row').count(), 8, '6 catalog entries + 2 unclaimed directories');
      assert.equal(await page.locator('#panel img').count(), 0, 'model names are rendered as text');
      await modelRow('<img onerror=alert(1)>').waitFor();
      assert.equal(await modelRow('onnx-whisper').count(), 0, 'stt/ lies outside the roots');
      assert.match(await page.locator('.mo-roots').innerText(), /embedding/);
      assert.match(await modelRow('ruri-v3-310m').first().innerText(), /ロード中/);
      assert.match(await modelRow('ruri-v3-reranker-310m').innerText(), /未取得/);
      await shot('models-desktop');
    });

    await check('models: row opens the drawer, a failed download blocks loading, Escape closes', async () => {
      await modelRow('synthetic-reranker').click();
      const drawer = page.locator('.mo-drawer');
      await drawer.waitFor();
      await drawer.getByText('取得が失敗', {exact: false}).first().waitFor();
      assert.equal(await drawer.getByRole('button', {name: 'ロード', exact: true}).isDisabled(), true);
      await shot('drawer-desktop');
      await page.keyboard.press('Escape');
      await drawer.waitFor({state: 'hidden'});
    });

    await check('models: ロード / 解放 round trip from the row action', async () => {
      const qwen = modelRow('Qwen3-Embedding-0.6B');
      await qwen.getByRole('button', {name: 'ロード', exact: true}).click();
      await qwen.getByRole('button', {name: '解放', exact: true}).waitFor();
      assert.ok(posts.some(p => p.url === '/admin/models/embedding/Qwen3-Embedding-0.6B/load'));
      await qwen.getByRole('button', {name: '解放', exact: true}).click();
      await qwen.getByRole('button', {name: 'ロード', exact: true}).waitFor();
      await qwen.getByRole('button', {name: 'ロード', exact: true}).click();
      await qwen.getByRole('button', {name: '解放', exact: true}).waitFor();
    });

    await check('models: catalog download under the first root reaches ロード via the feed', async () => {
      const entry = modelRow('Qwen3-Embedding-4B');
      await entry.getByRole('button', {name: '取得', exact: true}).click();
      await entry.getByRole('button', {name: 'ロード', exact: true}).waitFor({timeout: 15000});
      const download = posts.find(p => p.url === '/admin/models/download');
      assert.deepEqual(JSON.parse(download.body), {repo_id: 'Qwen/Qwen3-Embedding-4B', local_name: 'embedding/Qwen3-Embedding-4B', force: false});
      await shot('download-desktop');
    });

    await check('old ?tab= values land on the overview', async () => {
      for (const tab of ['catalog', 'downloads']) {
        await page.goto(`${origin}/admin/ui?tab=${tab}`);
        await modelRow('Qwen3-Embedding-4B').waitFor();
        assert.equal(new URL(page.url()).searchParams.get('tab'), 'models');
      }
    });

    await check('playground: dimension error is shown, then vectors and cosine similarity', async () => {
      await page.getByRole('button', {name: SIDEBAR[1]}).click();
      // The hub ticks once a second, so the just-loaded model may arrive one snapshot later.
      await page.locator('#pg-model option', {hasText: 'embedding/Qwen3-Embedding-0.6B'}).waitFor({state: 'attached'});
      assert.deepEqual(await page.locator('#pg-model option').allTextContents(), ['embedding/Qwen3-Embedding-0.6B', 'embedding/ruri-v3-310m']);
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
      await page.getByRole('button', {name: SIDEBAR[2]}).click();
      await row('#obs-table', 'embedding').first().waitFor();
      assert.deepEqual(await page.locator('#obs-table th').allTextContents(), ['Time', 'Event', 'Model', 'Status', 'Duration']);
      await page.locator('#obs-table .status-fail').first().waitFor();
      await page.getByRole('button', {name: SIDEBAR[3]}).click();
      await page.locator('#health-body', {hasText: 'Synthetic GPU'}).waitFor();
      assert.match(await page.locator('#health-body').innerText(), /推論確認/);
      assert.match(await page.locator('#health-body').innerText(), /Chat proxy/);
      await shot('health-desktop');
    });

    await check('mobile layout has no horizontal overflow', async () => {
      await page.setViewportSize({width: 390, height: 844});
      for (const id of ['models', 'playground']) {
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
      await page.goto(origin + '/prefix/admin/ui?p=1#models');
      await modelRow('Qwen3-Embedding-4B').waitFor();
    });

    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-qa.json'), JSON.stringify({fixture: true, production_inference: false, checks, page_errors: errors}, null, 2));
    console.log('Browser QA passed:', checks.length, 'scenarios');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
