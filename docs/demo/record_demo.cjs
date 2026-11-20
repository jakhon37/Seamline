// The README's demo: the real app on the demo project, driven by a script and filmed from the Xvfb display at
// 30 fps. The cursor and the keys pressed are drawn on top of the page, following the script's own mouse and keys.
// Usage: node docs/demo/record_demo.cjs PROJECT PROFILE PLAN.json OUT.mp4
const { createRequire } = require('module')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')
const APP = path.resolve(__dirname, '../../app')
const { _electron: electron } = createRequire(path.join(APP, 'package.json'))('@playwright/test')

const [project, profile, planFile, out] = process.argv.slice(2)
const plan = JSON.parse(fs.readFileSync(planFile, 'utf8'))
const W = 1440, H = 900
const pause = (page, ms) => page.waitForTimeout(ms)

// An arrow that follows the mouse, and a pill naming the key pressed, drawn above everything and never in the way
const OVERLAY = `(() => {
  const css = document.createElement('style')
  css.textContent = '#demo-cursor{position:fixed;left:0;top:0;width:22px;height:26px;z-index:2147483647;pointer-events:none;' +
    'transform:translate(-100px,-100px)} #demo-key{position:fixed;left:50%;top:72px;transform:translateX(-50%);z-index:2147483647;' +
    'pointer-events:none;display:flex;align-items:center;gap:10px;padding:8px 16px 8px 8px;border-radius:12px;background:#111317ee;' +
    'border:1px solid #2e333b;color:#e7e9ee;font:600 17px Inter Variable,sans-serif;opacity:0;transition:opacity .15s}' +
    '#demo-key b{min-width:32px;height:32px;padding:0 9px;border-radius:7px;background:#4f8cff;color:#fff;display:grid;place-items:center;font-weight:700}'
  document.head.appendChild(css)
  const cursor = document.createElement('div')
  cursor.id = 'demo-cursor'
  cursor.innerHTML = '<svg viewBox="0 0 22 26" width="22" height="26"><path d="M2 2 L2 21 L7.5 16 L11 24 L14.5 22.5 L11 15 L18.5 15 Z" ' +
    'fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>'
  document.body.appendChild(cursor)
  const key = document.createElement('div')
  key.id = 'demo-key'
  document.body.appendChild(key)
  addEventListener('mousemove', (e) => { cursor.style.transform = 'translate(' + e.clientX + 'px,' + e.clientY + 'px)' }, true)
  window.demoKey = (k, what) => {
    key.innerHTML = '<b>' + k + '</b>' + what
    key.style.opacity = 1
    clearTimeout(window.demoKeyTimer)
    window.demoKeyTimer = setTimeout(() => { key.style.opacity = 0 }, 1100)
  }
})()`

async function glide(page, to, ms = 600) {
  const from = await page.evaluate(() => {
    const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(document.getElementById('demo-cursor').style.transform)
    return m ? [parseFloat(m[1]), parseFloat(m[2])] : [720, 450]
  })
  const steps = Math.max(8, Math.round(ms / 16))
  for (let i = 1; i <= steps; i++) {
    const t = i / steps, e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2
    await page.mouse.move(from[0] + (to[0] - from[0]) * e, from[1] + (to[1] - from[1]) * e)
    await pause(page, ms / steps)
  }
}
async function center(page, locator) {
  const b = await locator.boundingBox()
  return [b.x + b.width / 2, b.y + b.height / 2]
}
async function click(page, locator, ms = 600) {
  await glide(page, await center(page, locator), ms)
  await pause(page, 150)
  await locator.click()
}
async function key(page, keys, label, what) {
  await page.evaluate(([k, w]) => window.demoKey(k, w), [label, what])
  await pause(page, 250)
  await page.keyboard.press(keys)
}

;(async () => {
  const app = await electron.launch({ args: [APP, `--user-data-dir=${profile}`], env: { ...process.env, MYSEG_PROJECT: project } })
  const page = await app.firstWindow()
  await app.evaluate(({ BrowserWindow }, [w, h]) => {
    const win = BrowserWindow.getAllWindows()[0]
    win.setContentBounds({ x: 0, y: 0, width: w, height: h })
    win.focus()
  }, [W, H])
  await page.locator('.file').first().waitFor({ timeout: 60000 })
  // Off camera: the estimate engine loads and both sizes are estimated, so the film shows them at once
  await page.keyboard.press('Control+2')
  await page.getByTestId('params').filter({ hasText: /M$/ }).waitFor({ timeout: 120000 })
  await page.getByRole('radio', { name: 'Fast' }).click()
  await page.getByTestId('params').filter({ hasText: '14.3 M' }).waitFor({ timeout: 60000 })
  await page.getByRole('radio', { name: 'Balanced' }).click()
  await page.getByTestId('params').filter({ hasText: '24.4 M' }).waitFor({ timeout: 60000 })
  await page.keyboard.press('Control+1')
  await page.evaluate(OVERLAY)
  // The starting view: images to review, least certain first, on the first one
  await page.locator('.filters button', { hasText: 'Review' }).click()
  await page.getByLabel('Order').selectOption('certainty')
  await page.locator('.file', { hasText: plan.first }).click()
  await page.mouse.move(W / 2, H / 2)
  await pause(page, 2500)

  const film = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'x11grab', '-draw_mouse', '0', '-framerate', '30',
    '-video_size', `${W}x${H}`, '-i', `${process.env.DISPLAY}.0+0,0`, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '10',
    '-pix_fmt', 'yuv444p', out], { stdio: ['pipe', 'inherit', 'inherit'] })
  await pause(page, 1500)

  for (const step of plan.steps) {
    if (step.key) await key(page, step.key, step.label, step.what)
    if (step.image) await click(page, page.locator('.file', { hasText: step.image }), step.ms)
    if (step.text) await click(page, page.getByText(step.text, { exact: true }).first(), step.ms)
    if (step.role) await click(page, page.getByRole(step.role[0], { name: step.role[1] }).first(), step.ms)
    if (step.run) await click(page, page.locator('.compare tbody tr', { hasText: step.run }), step.ms)
    if (step.move) {
      const box = await page.getByTestId('canvas').boundingBox()
      await glide(page, [box.x + step.move[0], box.y + step.move[1]], step.ms ?? 700)
    }
    if (step.stroke) {
      const box = await page.getByTestId('canvas').boundingBox()
      const [a, b] = step.stroke
      await glide(page, [box.x + a[0], box.y + a[1]], 500)
      await page.mouse.down()
      await glide(page, [box.x + b[0], box.y + b[1]], step.ms ?? 900)
      await page.mouse.up()
    }
    await pause(page, step.wait ?? 1500)
  }
  await pause(page, 800)
  film.stdin.write('q')
  await new Promise((done) => film.on('exit', done))
  await app.close()
  console.log('filmed', out)
})().catch((e) => { console.error(e); process.exit(1) })
