// The README's stills, taken from the real app on the demo project, and the pre-labelling they show.
// Usage: node docs/demo/shoot.cjs prelabel PROJECT PROFILE
//        node docs/demo/shoot.cjs stills PROJECT PROFILE PLAN.json OUTDIR [shot ...]
const { createRequire } = require('module')
const fs = require('fs')
const path = require('path')
const APP = path.resolve(__dirname, '../../app')
const { _electron: electron } = createRequire(path.join(APP, 'package.json'))('@playwright/test')

const [mode, project, profile, planFile, outDir, ...only] = process.argv.slice(2)
const W = 1440, H = 900

async function open(scale = 1) {
  const args = [APP, `--user-data-dir=${profile}`]
  const app = await electron.launch({ args, env: { ...process.env, MYSEG_PROJECT: project } })
  const page = await app.firstWindow()
  if (scale === 1) {
    await page.setViewportSize({ width: W, height: H }).catch(() => {})
  } else {
    // Playwright's viewport keeps one pixel to a point: ask the page itself for the larger scale
    const cdp = await page.context().newCDPSession(page)
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: scale, mobile: false })
    page.devicePixels = async (file) => {
      const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false })
      fs.writeFileSync(file, Buffer.from(data, 'base64'))
    }
  }
  await page.locator('.file').first().waitFor({ timeout: 60000 })
  return { app, page }
}

// The list as the README shows it: a filter, an order, an image
async function show(page, { filter, order, image, review }) {
  await page.keyboard.press('Control+1')
  if (filter) await page.locator('.filters button', { hasText: filter }).click()
  if (order) await page.getByLabel('Order').selectOption(order)
  if (image) await page.locator('.file', { hasText: image }).click()
  await page.waitForTimeout(1500)
  // W starts the review: the least certain shape, selected and in view
  if (review) {
    await page.keyboard.press('w')
    await page.waitForTimeout(800)
  }
}

// Zoom around a point on the canvas to a scale, as the wheel does
async function zoomTo(page, scale, at) {
  const canvas = page.getByTestId('canvas')
  const box = await canvas.boundingBox()
  const x = box.x + (at ? at[0] : box.width / 2), y = box.y + (at ? at[1] : box.height / 2)
  await page.mouse.move(x, y)
  for (let i = 0; i < 6; i++) {
    const now = parseFloat((await page.locator('.status').innerText()).match(/([\d.]+)%/)[1]) / 100
    if (Math.abs(now - scale) / scale < 0.004) break
    await page.mouse.wheel(0, -Math.log(scale / now) / 0.0022)
    await page.waitForTimeout(300)
  }
}

async function still(page, file, scale = 1) {
  await page.mouse.move(W - 4, H - 4)
  await page.waitForTimeout(1200)
  if (page.devicePixels) await page.devicePixels(file)
  else await page.screenshot({ path: file })
  console.log('saved', file, scale)
}

const SHOTS = {
  // Hand-drawn polygons, one selected
  async label(page, s) {
    await show(page, { filter: 'Done', order: 'name', image: s.image })
    await page.keyboard.press('0')
    await page.locator('.shape').nth(s.shape ?? 0).click()
    // Whole rows in the list, none cut at its top
    await page.locator('.files').evaluate((el) => {
      const row = el.querySelector('.file').getBoundingClientRect().height + 1
      el.scrollTop = Math.round(el.scrollTop / row) * row
    })
    await still(page, path.join(outDir, 'label.png'))
  },
  // A proposal in review, focused as the app focuses it
  async review(page, s) {
    await show(page, { filter: 'Review', order: 'certainty', image: s.image, review: true })
    if (s.zoom) await zoomTo(page, s.zoom, s.at)
    await still(page, path.join(outDir, 'review.png'))
  },
  // The brush partway through a stroke that joins two proposed pieces
  async brush(page, s) {
    await show(page, { filter: 'Review', order: 'certainty', image: s.image, review: true })
    if (s.zoom) await zoomTo(page, s.zoom, s.at)
    await page.keyboard.press('b')
    if (s.probe) {
      await page.mouse.move(W - 4, H - 4)
      await page.waitForTimeout(800)
      await page.screenshot({ path: path.join(outDir, 'brush-probe.png') })
      return
    }
    const box = await page.getByTestId('canvas').boundingBox()
    await page.mouse.move(box.x + s.from[0], box.y + s.from[1])
    await page.mouse.down()
    const steps = 12
    for (let i = 1; i <= steps; i++) {
      await page.mouse.move(box.x + s.from[0] + ((s.to[0] - s.from[0]) * i) / steps, box.y + s.from[1] + ((s.to[1] - s.from[1]) * i) / steps)
      await page.waitForTimeout(30)
    }
    await page.waitForTimeout(800)
    await page.screenshot({ path: path.join(outDir, 'brush.png') })
    console.log('saved brush')
    await page.mouse.up()
  },
  // Training here or over SSH, two real runs compared, and the model's costs
  async train(page, s) {
    await page.keyboard.press('Control+2')
    await page.waitForTimeout(1500)
    if (s.run) await page.locator('.run', { hasText: s.run }).click()
    // The form as the run had it, as Duplicate puts it back
    if (s.duplicate) await page.getByRole('button', { name: 'Duplicate' }).click()
    if (s.compare) await page.getByRole('button', { name: 'Compare runs' }).click()
    if (s.timeThere) {
      await page.getByRole('button', { name: /Time a tile on/ }).click()
      await page.getByTestId('time-there').filter({ hasText: /ms/ }).waitFor({ timeout: 300000 })
    }
    await page.getByTestId('params').filter({ hasText: /M$/ }).waitFor({ timeout: 120000 })
    await page.locator('pre[data-testid="log"]').evaluate((el) => el.scrollTo(0, el.scrollHeight))
    await still(page, path.join(outDir, 'train.png'))
  }
}

;(async () => {
  if (mode === 'prelabel') {
    const { app, page } = await open()
    await show(page, {})
    const button = page.getByRole('button', { name: /^Pre-label \d+ images$/ })
    console.log(await button.innerText())
    await button.click()
    await page.getByText(/Pre-labelled \d+ of \d+ images/).waitFor({ timeout: 1800000 })
    console.log(await page.getByText(/Pre-labelled \d+ of \d+ images/).innerText())
    await app.close()
    return
  }
  const plan = JSON.parse(fs.readFileSync(planFile, 'utf8'))
  fs.mkdirSync(outDir, { recursive: true })
  const wanted = only.length ? only : Object.keys(plan)
  for (const name of wanted.filter((n) => n !== 'hero')) {
    const { app, page } = await open()
    await SHOTS[name](page, plan[name])
    await app.close()
  }
  if (wanted.includes('hero')) {
    // 2000 px wide: the same window drawn at a larger scale
    const s = plan.hero
    const { app, page } = await open(2000 / W)
    await show(page, { filter: 'Review', order: 'certainty', image: s.image, review: true })
    if (s.fit) await page.keyboard.press('0')
    if (s.zoom) await zoomTo(page, s.zoom, s.at)
    if (s.shape !== undefined) await page.locator('.shape').nth(s.shape).click()
    await still(page, path.join(outDir, 'hero.png'), 2000 / W)
    await app.close()
  }
})().catch((e) => { console.error(e); process.exit(1) })
