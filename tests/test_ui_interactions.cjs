const {test, before, after, beforeEach, afterEach} = require('node:test');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const origin = 'http://127.0.0.1:19975';
let browser, context, page, posts, errors;
const refresh = async () => {
  await page.locator('#refresh').click();
  await page.waitForFunction(() => !document.getElementById('refresh').disabled);
};
const chooseUnloaded = async () => {
  await page.locator('#model-search').fill('Qwen3-Embedding-0.6B');
  await page.getByRole('button', {name:'ロード', exact:true}).click();
  await page.locator('#confirm-dialog').waitFor();
};

before(async () => {
  browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH, headless:true, args:['--no-sandbox']});
});
after(async () => { await browser?.close(); });
beforeEach(async () => {
  context = await browser.newContext({viewport:{width:390,height:844}});
  await context.request.post(`${origin}/fixture/reset`);
  page = await context.newPage(); posts = []; errors = [];
  page.on('request', request => { if (request.method() === 'POST') posts.push(request); });
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${origin}/ui`);
  await page.getByText('接続済み', {exact:true}).waitFor();
});
afterEach(async () => { await context?.close(); assert.deepEqual(errors, []); });

test('repeated cancel and Escape never submit load or unload', {timeout:30000}, async () => {
  for (const escape of [false, true, false]) {
    await chooseUnloaded();
    assert.equal(await page.locator('#confirm-dialog [value="cancel"]').evaluate(n => n === document.activeElement), true);
    if (escape) await page.keyboard.press('Escape');
    else await page.getByRole('button', {name:'キャンセル', exact:true}).click();
    await page.locator('#confirm-dialog').waitFor({state:'hidden'});
  }
  await page.locator('#model-search').fill('embedding/ruri');
  await page.getByRole('button', {name:'アンロード', exact:true}).click();
  await page.keyboard.press('Escape');
  await page.locator('#confirm-dialog').waitFor({state:'hidden'});
  assert.equal(posts.length, 0);
  await page.getByText('ロード済み・推論未確認', {exact:true}).waitFor();
});

test('repeated confirmation creates one POST and gates navigation while pending', {timeout:30000}, async () => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const started = new Promise(resolve => page.route('**/v1/models/load', async route => {
    resolve(); await gate; await route.continue();
  }));
  await chooseUnloaded();
  await page.locator('#confirm-action').evaluate(button => { button.click(); button.click(); });
  await started;
  assert.equal(posts.length, 1);
  assert.equal(await page.locator('[data-page]').evaluateAll(nodes => nodes.every(n => n.disabled)), true);
  assert.equal(await page.getByRole('button', {name:'ロード', exact:true}).isDisabled(), true);
  release();
  await page.getByText('ロードが完了しました。推論はまだ確認していません。', {exact:true}).waitFor();
  await page.getByText('ロード済み・推論未確認', {exact:true}).waitFor();
  assert.equal(posts.length, 1);
});

test('a completed load with failed follow-up read reports success and recovers without another POST', {timeout:30000}, async () => {
  let failed = false;
  await page.route('**/health', route => failed
    ? route.fulfill({status:503,contentType:'application/json',body:'{"detail":"synthetic follow-up outage"}'}) : route.continue());
  await page.route('**/v1/models/load', async route => {
    const response = await route.fetch(); failed = true; await route.fulfill({response});
  });
  await chooseUnloaded(); await page.locator('#confirm-action').click();
  await page.locator('#notice').filter({hasText:'ロードが完了しました。推論はまだ確認していません。 状態の再取得に失敗'}).waitFor();
  await page.getByText('取得失敗・操作停止', {exact:true}).waitFor();
  assert.equal(posts.length, 1);
  failed = false; await refresh();
  await page.getByText('接続済み', {exact:true}).waitFor();
  await page.getByText('ロード済み・推論未確認', {exact:true}).waitFor();
  assert.equal(posts.length, 1);
});

test('load failure survives refresh, then an explicit retry succeeds once', {timeout:30000}, async () => {
  await page.route('**/v1/models/load', route => route.fulfill({status:500,contentType:'application/json',body:'{"detail":"synthetic load failure"}'}));
  await chooseUnloaded(); await page.locator('#confirm-action').click();
  await page.locator('#error').filter({hasText:'synthetic load failure'}).waitFor();
  await refresh(); assert.match(await page.locator('#error').textContent(), /synthetic load failure/);
  await page.unroute('**/v1/models/load');
  await chooseUnloaded(); await page.locator('#confirm-action').click();
  await page.getByText('ロード済み・推論未確認', {exact:true}).waitFor();
  assert.equal(posts.length, 2);
  assert.equal(await page.locator('#error').isVisible(), false);
});

test('repeated playground submission is bounded and a malformed vector can be retried', {timeout:30000}, async () => {
  await page.locator('[data-page="playground"]').click();
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const started = new Promise(resolve => page.route('**/v1/embeddings', async route => {
    resolve(); await gate;
    await route.fulfill({status:200,contentType:'application/json',body:'{"model":"synthetic","data":[{"index":0,"embedding":[1]},{"index":1,"embedding":[]}]}'});
  }));
  await page.locator('#test-form').evaluate(form => {
    form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
    form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
  });
  await started; assert.equal(posts.length, 1);
  assert.equal(await page.locator('#test-button').isDisabled(), true);
  release();
  await page.locator('#test-validation').filter({hasText:'有効なベクトル'}).waitFor();
  assert.equal(await page.locator('#test-result').textContent(), '');
  await page.unroute('**/v1/embeddings');
  await page.locator('#test-button').click();
  await page.getByText('ベクトル生成が完了しました', {exact:true}).waitFor();
  assert.equal(posts.length, 2);
  assert.equal(await page.locator('#test-validation').textContent(), '');
});

test('repeated refresh requests coalesce while forms and the next refresh remain usable', {timeout:30000}, async () => {
  await page.locator('[data-page="playground"]').click();
  await page.locator('#text-a').fill('synthetic retained text');
  let release, reads = 0;
  const gate = new Promise(resolve => { release = resolve; });
  const started = new Promise(resolve => page.route('**/health', async route => {
    reads++; resolve(); await gate; await route.continue();
  }));
  await page.locator('#refresh').evaluate(button => {
    button.dispatchEvent(new Event('click'));
    button.dispatchEvent(new Event('click'));
    button.dispatchEvent(new Event('click'));
  });
  await started; assert.equal(reads, 1);
  release(); await page.waitForFunction(() => !document.getElementById('refresh').disabled);
  assert.equal(await page.locator('#text-a').inputValue(), 'synthetic retained text');
  await refresh(); assert.equal(reads, 2);
  assert.equal(posts.length, 0);
});
