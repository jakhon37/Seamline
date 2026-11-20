// Train the README's demo runs through the app on a GPU machine over SSH: Fast, then Balanced, so Balanced is the
// newest run and the one pre-labelling uses. A run already finished is skipped, and one still training on the GPU
// is waited for, so a stop on this side picks up where it left off. Runs on a display of its own, such as Xvfb.
// Usage: node docs/demo/train_runs.cjs PROJECT PROFILE LOG HOST DIR
const { createRequire } = require('module')
const fs = require('fs')
const path = require('path')
const APP = path.resolve(__dirname, '../../app')
const { _electron: electron } = createRequire(path.join(APP, 'package.json'))('@playwright/test')

const [project, profile, logFile, host, dir] = process.argv.slice(2)
const log = (line) => fs.appendFileSync(logFile, `${new Date().toISOString().slice(11, 19)} ${line}\n`)
const SETTINGS = { epochs: '100', batch: '8', size: '512', lr: '0.0003', val: '0.2', tile: '1024' }
const RUNS = [{ size: 'Fast', encoder: 'resnet18' }, { size: 'Balanced', encoder: 'resnet34' }]

function finished(encoder) {
  const runs = path.join(project, '.myseg/runs')
  if (!fs.existsSync(runs)) return false
  return fs.readdirSync(runs).some((name) => {
    const options = path.join(runs, name, 'options.json')
    const events = path.join(runs, name, 'events.jsonl')
    if (!fs.existsSync(options) || !fs.existsSync(events)) return false
    const done = fs.readFileSync(events, 'utf8').includes('"type": "done"')
    const model = fs.readdirSync(path.join(runs, name)).some((f) => f.startsWith('run_'))
    return JSON.parse(fs.readFileSync(options, 'utf8')).encoder === encoder && done && model
  })
}

async function untilDone(page, what) {
  const stop = page.getByRole('button', { name: /Stop training/ })
  let last = ''
  for (;;) {
    if (!(await stop.count())) return
    const epoch = await page.locator('.tile', { hasText: 'Epoch' }).innerText().catch(() => '')
    const line = epoch.replace(/\s+/g, ' ')
    if (line !== last) log(`${what}: ${line}`)
    last = line
    await page.waitForTimeout(30000)
  }
}

;(async () => {
  for (const run of RUNS) {
    if (finished(run.encoder)) { log(`${run.size} already trained`); continue }
    const app = await electron.launch({
      args: [APP, `--user-data-dir=${profile}`],
      env: { ...process.env, MYSEG_PROJECT: project }
    })
    const page = await app.firstWindow()
    await page.setViewportSize({ width: 1440, height: 900 }).catch(() => {})
    await page.getByRole('tab', { name: /Train/ }).click()
    await page.waitForTimeout(5000)
    // A run that outlived the app reattaches when its folder opens
    if (await page.getByRole('button', { name: /Stop training/ }).count()) {
      log(`${run.size}: reattached to a run in progress`)
      await untilDone(page, run.size)
    } else {
      await page.getByLabel('Run on').selectOption('ssh')
      await page.getByLabel('SSH host').fill(host)
      await page.getByLabel('Remote folder').fill(dir)
      await page.getByRole('radio', { name: run.size }).click()
      await page.getByLabel('Epochs', { exact: true }).fill(SETTINGS.epochs)
      await page.getByLabel('Batch').fill(SETTINGS.batch)
      await page.getByLabel('Input size').fill(SETTINGS.size)
      await page.getByLabel('Learning rate').fill(SETTINGS.lr)
      await page.getByLabel('Validation share').fill(SETTINGS.val)
      await page.getByLabel('Tile size').fill(SETTINGS.tile)
      await page.getByLabel('Start from ImageNet weights').check()
      log(`${run.size}: starting`)
      await page.getByRole('button', { name: /Train on \d+ labelled/ }).click()
      await page.getByRole('button', { name: /Stop training/ }).waitFor({ timeout: 900000 })
      log(`${run.size}: training on ${host}`)
      await untilDone(page, run.size)
    }
    // The model comes back and the remote copy goes before the Train button returns
    await page.getByRole('button', { name: /Train on \d+ labelled/ }).waitFor({ timeout: 600000 })
    const status = await page.locator('.tile', { hasText: 'Status' }).innerText()
    log(`${run.size}: ${status.replace(/\s+/g, ' ')}; trained: ${finished(run.encoder)}`)
    await app.close()
  }
  log('all runs trained')
})().catch((e) => { log(`failed: ${e.message.split('\n')[0]}`); process.exit(1) })
