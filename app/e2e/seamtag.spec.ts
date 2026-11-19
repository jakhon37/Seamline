import { expect, test } from '@playwright/test'
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import path from 'path'
import { classes, copy, drag, fakeEngine, fakeSeamtag, finish, folder, image, launch, open, read, shot } from './seamtag_folder'

const SIZE = [400, 300]
const seamtagButton = (page: import('@playwright/test').Page) => page.locator('.topbar').getByRole('button', { name: 'Label in Seamtag' })

test('Label in Seamtag starts Seamtag on the image, even one in a subfolder', async () => {
  const dir = folder('launch')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  await image(dir, 'sub/inner.png', SIZE[0], SIZE[1])
  await image(dir, 'boxes.png', 64, 48)
  copy(dir, 'voc_cars.xml', 'boxes.xml')
  const seamtag = fakeSeamtag(folder('launch-seamtag'), path.join(dir, 'top.json'))
  const { app, page, errors } = await launch(dir, { SEAMLINE_SEAMTAG: seamtag.script })

  // A change still waiting to be saved is saved first, so Seamtag opens it
  await open(page, 'top.png', SIZE)
  await page.keyboard.press('b')
  await drag(page, SIZE, [100, 100], [200, 100])
  await seamtagButton(page).click()
  await expect.poll(seamtag.args).toEqual([path.join(dir, 'top.png'), 'labels saved'])
  expect(read(dir, 'top.json').shapes).toHaveLength(1)

  // An image in a subfolder goes as itself too: Seamtag 3.3 opens the whole folder on it
  await open(page, 'sub/inner.png', SIZE)
  await seamtagButton(page).click()
  await expect.poll(() => seamtag.args().slice(2)).toEqual([path.join(dir, 'sub/inner.png'), 'labels saved'])

  // The same from the bar on an image whose labels Seamtag keeps
  await open(page, 'boxes.png', [64, 48])
  await page.getByTestId('read-only-bar').getByRole('button', { name: 'Label in Seamtag' }).click()
  await expect.poll(() => seamtag.args().slice(4)).toEqual([path.join(dir, 'boxes.png'), 'labels saved'])
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test("with no Seamtag found, it says so and offers the download page or the program you pick", async () => {
  const dir = folder('missing')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  await image(dir, 'next.png', SIZE[0], SIZE[1])
  const seamtag = fakeSeamtag(folder('missing-seamtag'))
  const { app, page, errors, home } = await launch(dir)
  await open(page, 'top.png', SIZE)
  await page.keyboard.press('b')
  await drag(page, SIZE, [100, 100], [200, 100])
  await expect(page.locator('.shape')).toHaveCount(1)
  await page.locator('.shape').first().click()
  const dialog = page.getByRole('dialog')

  await seamtagButton(page).click()
  await expect(dialog).toContainText("Seamline can't find Seamtag")
  await shot(page, 'seamtag-missing')
  // Keys go to the dialog, not to the image behind it
  await expect(dialog.getByRole('button', { name: 'Cancel' })).toBeFocused()
  for (const key of ['Delete', 'd']) await page.keyboard.press(key)
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'top.png')
  await expect(page.locator('.labeler .shape')).toHaveCount(1)
  await page.waitForTimeout(600)
  expect(read(dir, 'top.json').shapes).toHaveLength(1)
  // The page opens only when asked for, in the browser
  await app.evaluate(({ shell }) => {
    (globalThis as any).opened = [] as string[]
    shell.openExternal = (async (url: string) => { (globalThis as any).opened.push(url) }) as typeof shell.openExternal
  })
  await dialog.getByRole('button', { name: 'Download Seamtag' }).click()
  await expect(dialog).toHaveCount(0)
  expect(await app.evaluate(() => (globalThis as any).opened)).toEqual(['https://github.com/jakhon37/Seamtag/releases/latest'])
  expect(seamtag.args()).toEqual([])

  // Located by hand, it starts at once, and is remembered
  await seamtagButton(page).click()
  await app.evaluate(({ dialog: d }, program) => {
    d.showOpenDialog = (async () => ({ canceled: false, filePaths: [program] })) as any
  }, seamtag.script)
  await dialog.getByRole('button', { name: 'Locate Seamtag…' }).click()
  await expect(dialog).toHaveCount(0)
  await expect.poll(seamtag.args).toEqual([path.join(dir, 'top.png')])
  const profile = await app.evaluate(({ app: a }) => a.getPath('userData'))
  expect(profile.startsWith(home)).toBe(true)
  expect(JSON.parse(readFileSync(path.join(profile, 'seamtag.json'), 'utf8'))).toEqual({ path: seamtag.script })
  await seamtagButton(page).click()
  await expect.poll(seamtag.args).toEqual([path.join(dir, 'top.png'), path.join(dir, 'top.png')])
  await expect(dialog).toHaveCount(0)
  await finish(app, errors)
})

test("while Seamtag starts, its buttons say so and can't start another", async () => {
  const dir = folder('starting')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  // A run with an exported model, so the Train page offers Open in Seamtag
  const run = path.join(dir, '.myseg/runs/20260101_000000')
  mkdirSync(path.join(run, 'run_1'), { recursive: true })
  const files = [
    ['run_1/best_model.pth', 'weights'], ['model.onnx', 'onnx'], ['events.jsonl', '{"type": "done"}\n'],
    ['options.json', '{"architecture": "Unet", "encoder": "resnet18"}']
  ]
  for (const [name, text] of files) {
    writeFileSync(path.join(run, name), text)
  }
  // It keeps running, as Seamtag does: Seamline waits 2 s to see it doesn't close at once
  const seamtag = fakeSeamtag(folder('starting-seamtag'), '', 6)
  const { app, page, errors } = await launch(dir, { SEAMLINE_SEAMTAG: seamtag.script })
  const top = page.locator('.topbar .ghost', { hasText: 'Seamtag' })
  const card = page.getByTestId('export').locator('.ghost', { hasText: /Open in|Starting/ })

  await page.getByRole('tab', { name: /Train/ }).click()
  await card.click()
  for (const button of [card, top]) {
    await expect(button).toHaveText('Starting Seamtag…')
    await expect(button).toBeDisabled()
  }
  await shot(page, 'seamtag-starting')
  await card.click({ force: true })
  await top.click({ force: true })
  await expect(card).toHaveText('Open in Seamtag')
  await expect(top).toHaveText('Label in Seamtag')
  expect(seamtag.args()).toEqual([dir])

  await page.getByRole('tab', { name: /Label/ }).click()
  await open(page, 'top.png', SIZE)
  await top.click()
  await expect(top).toHaveText('Starting Seamtag…')
  await expect(top).toBeDisabled()
  await top.click({ force: true })
  await expect(top).toHaveText('Label in Seamtag')
  await expect(top).toBeEnabled()
  expect(seamtag.args()).toEqual([dir, path.join(dir, 'top.png')])
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

// Called as: python engine/studio.py COMMAND PROJECT [--run RUN | --val-split V --tile T]
const ENGINE = `
const fs = require('fs'), path = require('path')
const [, , , command, root, , run] = process.argv
if (command === 'export') {
  const dir = path.join(root, '.myseg/runs', run)
  if (!fs.existsSync(path.join(dir, 'run_1/best_model.pth')) || run.endsWith('3_000000')) {
    console.error('Traceback (most recent call last):')
    console.error('Run ' + run + ' has no saved model yet.')
    process.exit(1)
  }
  setTimeout(() => {
    fs.writeFileSync(path.join(dir, 'model.onnx'), 'onnx')
    fs.writeFileSync(path.join(dir, 'model.json'), JSON.stringify({ classes: ['cat'] }))
    console.log('Exporting to ONNX')
    console.log(JSON.stringify({ model: path.join(dir, 'model.onnx'), sidecar: path.join(dir, 'model.json'), classes: ['cat'], input_size: 512, tile: 1024 }))
  }, 800)
} else if (command === 'prepare') {
  const known = JSON.parse(fs.readFileSync(path.join(root, '.myseg/project.json'), 'utf8')).classes.map((c) => c.name)
  console.log(JSON.stringify({
    labelled: 3, unlabelled: 1, train: 2, val: 1,
    unknown_labels: ['bus', 'tree'].filter((l) => !known.includes(l)),
    class_pixels: Object.fromEntries(['background', ...known].map((c) => [c, 1000])),
    skipped_shapes: { line: 3, point: 1 },
    formats: { labelme: 2, voc: 0, yolo: 1, createml: 0 }
  }))
} else {
  process.exit(2)
}
`

test('Export for Seamtag runs the engine and says where the model went', async () => {
  const dir = folder('export')
  classes(dir, ['cat'])
  await image(dir, 'one.png', SIZE[0], SIZE[1])
  const run = (name: string, model: boolean) => {
    const at = path.join(dir, '.myseg/runs', name)
    mkdirSync(path.join(at, 'run_1'), { recursive: true })
    writeFileSync(path.join(at, 'events.jsonl'), [
      { type: 'start', epochs: 1, device: 'cpu' }, { type: 'epoch', epoch: 0, epochs: 1, train_loss: 0.5, val_loss: 0.5, iou: 0.5, f1: 0.5 }, { type: 'done' }
    ].map((e) => JSON.stringify(e) + '\n').join(''))
    writeFileSync(path.join(at, 'options.json'), JSON.stringify({ architecture: 'Unet', encoder: 'resnet18' }))
    if (model) writeFileSync(path.join(at, 'run_1/best_model.pth'), 'weights')
  }
  run('20260101_000000', true)
  run('20260102_000000', false)
  run('20260103_000000', true)
  const tools = folder('export-tools')
  const seamtag = fakeSeamtag(tools)
  const { app, page, errors } = await launch(dir, { SEAMLINE_SEAMTAG: seamtag.script, MYSEG_PYTHON: fakeEngine(tools, ENGINE) })
  await page.getByRole('tab', { name: /Train/ }).click()
  const exportCard = page.getByTestId('export')

  // The newest run comes first: its model can't be exported, and the reason shows
  await expect(page.locator('.run.on')).toContainText('20260103 000000')
  await exportCard.getByRole('button', { name: 'Export for Seamtag' }).click()
  await expect(page.locator('.error')).toHaveText('Export failed: Run 20260103_000000 has no saved model yet.')
  await page.locator('.error').click()

  // A run with no model has nothing to export
  await page.locator('.run', { hasText: '20260102 000000' }).click()
  await expect(exportCard).toHaveCount(0)

  await page.locator('.run', { hasText: '20260101 000000' }).click()
  await expect(exportCard).toContainText("Export this run's model as ONNX")
  await exportCard.getByRole('button', { name: 'Export for Seamtag' }).click()
  await expect(exportCard.getByRole('button', { name: 'Exporting…' })).toBeVisible()
  await expect(exportCard.locator('p')).toHaveText('Saved as .myseg/runs/20260101_000000/model.onnx. Seamtag 3.3 pre-labels with ' +
    'the newest exported run unless another model is chosen in AI → Model for pre-labelling.')
  expect(existsSync(path.join(dir, '.myseg/runs/20260101_000000/model.onnx'))).toBe(true)
  await shot(page, 'export')
  await exportCard.getByRole('button', { name: 'Open in Seamtag' }).click()
  await expect.poll(seamtag.args).toEqual([dir])

  // The dataset check: shapes with no area, the formats, and labels the class list lacks, which can be added
  await page.locator('.icon[title="Check dataset"]').click()
  const check = page.locator('.kv')
  await expect(check).toContainText('Left out, no area: 3 lines, 1 point')
  // Only the formats the folder has, apart from the row's name
  const formats = check.locator('div', { hasText: 'Formats' })
  await expect(formats.locator('b')).toHaveText('LabelMe JSON 2 · YOLO 1')
  const [name, value] = await Promise.all(['span', 'b'].map(async (t) => (await formats.locator(t).boundingBox())!))
  expect(name.x + name.width).toBeLessThan(value.x)
  await expect(check).toContainText('Ignored, not in class list: bus, tree')
  await shot(page, 'dataset-check')
  await check.getByRole('button', { name: 'Add as classes' }).click()
  await expect(check).not.toContainText('Ignored')
  await expect(check).toContainText('bus pixels')
  const project = read(dir, '.myseg/project.json')
  expect(project.classes).toEqual([
    { name: 'cat', color: '#4f8cff' }, { name: 'bus', color: '#ff6b6b' }, { name: 'tree', color: '#3ecf8e' }
  ])
  await page.getByRole('tab', { name: /Label/ }).click()
  await expect(page.locator('.class .name')).toHaveText(['cat', 'bus', 'tree'])
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test("a Seamtag that closes as it starts says why, and isn't remembered when picked", async () => {
  const dir = folder('failing')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  const tools = folder('failing-tools')
  // As Seamtag's AppImage does on Ubuntu 24.04, run without --no-sandbox
  const failing = path.join(tools, 'Seamtag-3.2.0-linux-x86_64.AppImage')
  writeFileSync(failing, '#!/bin/sh\nexit 1\n')
  chmodSync(failing, 0o755)
  const seamtag = fakeSeamtag(tools)
  const { app, page, errors, home } = await launch(dir, { SEAMLINE_SEAMTAG: failing })
  await open(page, 'top.png', SIZE)
  const dialog = page.getByRole('dialog')

  await seamtagButton(page).click()
  await expect(dialog.getByRole('heading')).toHaveText("Seamtag didn't start")
  await expect(dialog).toContainText(`${failing} closed right after starting (exit code 1).`)
  await expect(dialog).toContainText("Seamtag's AppImage needs --no-sandbox on Ubuntu 24.04 and later, so the .deb is the better install there.")
  await shot(page, 'seamtag-failed')
  await expect(page.locator('.error')).toHaveCount(0)

  // Picked by hand, it fails the same way, and isn't remembered
  const pick = (program: string) => app.evaluate(({ dialog: d }, file) => {
    d.showOpenDialog = (async () => ({ canceled: false, filePaths: [file] })) as any
  }, program)
  await pick(failing)
  await dialog.getByRole('button', { name: 'Locate Seamtag…' }).click()
  await expect(dialog).toContainText('exit code 1')
  const profile = await app.evaluate(({ app: a }) => a.getPath('userData'))
  expect(profile.startsWith(home)).toBe(true)
  expect(existsSync(path.join(profile, 'seamtag.json'))).toBe(false)
  // One that starts is
  await pick(seamtag.script)
  await dialog.getByRole('button', { name: 'Locate Seamtag…' }).click()
  await expect(dialog).toHaveCount(0)
  await expect.poll(seamtag.args).toEqual([path.join(dir, 'top.png')])
  expect(JSON.parse(readFileSync(path.join(profile, 'seamtag.json'), 'utf8'))).toEqual({ path: seamtag.script })
  await finish(app, errors)
})

test('a copy that closes as it starts gives way to the next one found', async () => {
  const dir = folder('next-copy')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  const tools = folder('next-copy-tools')
  const failing = path.join(tools, 'old-seamtag')
  writeFileSync(failing, '#!/bin/sh\nexit 3\n')
  chmodSync(failing, 0o755)
  const seamtag = fakeSeamtag(path.join(tools, 'picked'))
  const { app, page, errors } = await launch(dir, { SEAMLINE_SEAMTAG: failing })
  await open(page, 'top.png', SIZE)
  const dialog = page.getByRole('dialog')
  // Alone, it says why, with no word of the AppImage, which it isn't
  await seamtagButton(page).click()
  await expect(dialog).toContainText(`${failing} closed right after starting (exit code 3).`)
  await expect(dialog).not.toContainText('AppImage')
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  // With another copy to try, the one picked before, that one starts
  const profile = await app.evaluate(({ app: a }) => a.getPath('userData'))
  writeFileSync(path.join(profile, 'seamtag.json'), JSON.stringify({ path: seamtag.script }))
  await seamtagButton(page).click()
  await expect.poll(seamtag.args).toEqual([path.join(dir, 'top.png')])
  await expect(dialog).toHaveCount(0)
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test("a copy picked before that no longer starts gives way to an installed one, and is forgotten", async () => {
  const dir = folder('forgotten')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  const tools = folder('forgotten-tools')
  // As Seamtag's AppImage is on Ubuntu 24.04, and its .deb's program on the PATH
  const picked = path.join(tools, 'Seamtag-3.2.0-linux-x86_64.AppImage')
  writeFileSync(picked, '#!/bin/sh\nexit 1\n')
  chmodSync(picked, 0o755)
  const installed = fakeSeamtag(path.join(tools, 'bin'))
  // No SEAMLINE_SEAMTAG: the search runs as for anyone
  const { app, page, errors } = await launch(dir, { SEAMLINE_SEAMTAG: '', PATH: `${path.join(tools, 'bin')}${path.delimiter}${process.env.PATH}` })
  // Whatever happens, never the Seamtag installed on this computer: only programs made here start
  await app.evaluate((_, allowed) => {
    const children = (process as any).getBuiltinModule('child_process')
    const spawn = children.spawn
    ;(globalThis as any).started = []
    children.spawn = (program: string, ...rest: unknown[]) => {
      ;(globalThis as any).started.push(program)
      return spawn(program.startsWith(allowed) ? program : '/bin/false', ...rest)
    }
  }, tools)
  const profile = await app.evaluate(({ app: a }) => a.getPath('userData'))
  const choice = path.join(profile, 'seamtag.json')
  writeFileSync(choice, JSON.stringify({ path: picked }))
  await open(page, 'top.png', SIZE)
  await seamtagButton(page).click()
  await expect.poll(installed.args).toEqual([path.join(dir, 'top.png')])
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect.poll(() => existsSync(choice)).toBe(false)
  // From then on, the installed one starts first
  await seamtagButton(page).click()
  await expect.poll(installed.args).toEqual([path.join(dir, 'top.png'), path.join(dir, 'top.png')])
  expect(await app.evaluate(() => (globalThis as any).started)).toEqual([picked, installed.script, installed.script])
  await finish(app, errors)
})

test("Seamtag gets the environment Seamline started in, not what Seamline's Electron set for itself", async () => {
  const dir = folder('env')
  classes(dir, ['cat'])
  await image(dir, 'top.png', SIZE[0], SIZE[1])
  const tools = folder('env-tools')
  const script = path.join(tools, 'seamtag'), out = path.join(tools, 'env.json')
  writeFileSync(script, `#!${process.execPath}\nrequire('fs').writeFileSync(${JSON.stringify(out)}, JSON.stringify(process.env))\n`)
  chmodSync(script, 0o755)
  const { app, page, errors } = await launch(dir, {
    SEAMLINE_SEAMTAG: script, SEAMLINE_E2E: 'kept', XDG_CURRENT_DESKTOP: 'ubuntu:GNOME',
    NODE_ENV_ELECTRON_VITE: 'development', ELECTRON_ENTRY: 'out/main/index.js', ELECTRON_CLI_ARGS: '[]'
  })
  // As under electron-vite's dev server, whose page Seamtag would load
  await app.evaluate(() => { process.env.ELECTRON_RENDERER_URL = 'http://127.0.0.1:5173/' })
  await open(page, 'top.png', SIZE)
  await seamtagButton(page).click()
  await expect.poll(() => existsSync(out)).toBe(true)
  const env = JSON.parse(readFileSync(out, 'utf8'))
  expect(env.SEAMLINE_E2E).toBe('kept')
  // Seamline's Electron calls the desktop Unity for itself
  expect(env.XDG_CURRENT_DESKTOP).toBe('ubuntu:GNOME')
  const gone = ['ELECTRON_RENDERER_URL', 'NODE_ENV_ELECTRON_VITE', 'ELECTRON_ENTRY', 'ELECTRON_CLI_ARGS', 'ORIGINAL_XDG_CURRENT_DESKTOP', 'CHROME_DESKTOP']
  if (!process.env.GDK_BACKEND) gone.push('GDK_BACKEND')
  expect(gone.filter((key) => key in env)).toEqual([])
  await finish(app, errors)
})
