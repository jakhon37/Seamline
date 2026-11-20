import { expect, test } from '@playwright/test'
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { fakeGpu, launch } from './fake_gpu'

const BASE = (process.env.E2E_DIR || '/tmp/myseg-e2e') + '-model-size'
const SHOTS = process.env.E2E_SHOTS || BASE
const OLD = '20260101_000000'

// A folder of labelled discs, small enough to train on here in seconds, and a run from before runs measured themselves
async function project(dir: string) {
  mkdirSync(path.join(dir, '.myseg/runs', OLD), { recursive: true })
  writeFileSync(path.join(dir, '.myseg/project.json'), JSON.stringify({ name: 'discs', classes: [{ name: 'disc', color: '#4f8cff' }] }))
  for (let i = 0; i < 8; i++) {
    const x = 10 + 4 * i
    const svg = `<svg width="64" height="64"><rect width="64" height="64" fill="#3c4650"/><circle cx="${x + 12}" cy="32" r="12" fill="#e6b43c"/></svg>`
    await sharp(Buffer.from(svg)).png().toFile(path.join(dir, `disc_${i}.png`))
    const points = Array.from({ length: 12 }, (_, a) => [x + 12 + 12 * Math.cos(a / 12 * 6.283), 32 + 12 * Math.sin(a / 12 * 6.283)])
    writeFileSync(path.join(dir, `disc_${i}.json`), JSON.stringify({
      version: '5.5.0', flags: {}, imagePath: `disc_${i}.png`, imageData: null, imageWidth: 64, imageHeight: 64,
      shapes: [{ label: 'disc', shape_type: 'polygon', flags: {}, points }]
    }))
  }
  // As the app saved runs until now: the form's options of the time, and its epochs
  const old = path.join(dir, '.myseg/runs', OLD)
  writeFileSync(path.join(old, 'options.json'), JSON.stringify({
    architecture: 'Unet', encoder: 'resnet18', encoder_weights: 'imagenet', loss: 'combined', epochs: 2, batch_size: 8,
    image_size: 512, learning_rate: 0.0001, val_split: 0.2, tile: 1024, num_workers: 4
  }))
  const epoch = (n: number, iou: number) => ({ type: 'epoch', epoch: n, epochs: 2, train_loss: 0.5, val_loss: 0.4, lr: 0.0001, iou, f1: iou })
  writeFileSync(path.join(old, 'events.jsonl'), [{ type: 'start', epochs: 2, device: 'CPU' }, epoch(0, 0.5), epoch(1, 0.25), { type: 'done' }]
    .map((e) => JSON.stringify(e) + '\n').join(''))
  return dir
}

test('model sizes and their estimate, advanced settings, duplicating and comparing runs', async () => {
  // Trains a small model for real on the CPU
  test.setTimeout(600000)
  rmSync(BASE, { recursive: true, force: true })
  const gpu = fakeGpu(path.join(BASE, 'fake'))
  const dir = await project(path.join(BASE, 'discs'))
  // The real engine here, and the stand-in GPU machine over SSH
  const app = await launch(gpu, BASE, dir, { MYSEG_PYTHON: '' })
  const page = await app.firstWindow()
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.setViewportSize({ width: 1500, height: 940 }).catch(() => {})
  await page.getByRole('tab', { name: /Train/ }).click()
  await page.getByLabel('Run on').selectOption('local')
  const size = (name: string) => page.getByRole('radio', { name })
  const params = page.getByTestId('params')

  // Balanced to start: Unet on resnet34, sized, then timed on this computer
  await expect(size('Balanced')).toHaveAttribute('aria-checked', 'true')
  await expect(params).toHaveText('24.4 M', { timeout: 120000 })
  await expect(page.getByTestId('estimate')).toContainText(/GFLOPs/)
  await expect(page.getByTestId('estimate')).toContainText(/Memory at batch 8\s*≈ \d+\.\d GB/)
  await expect(page.getByTestId('time-here')).toHaveText(/^\d+(\.\d)? ms · CPU$/, { timeout: 120000 })
  await expect(page.getByTestId('weights')).toHaveText("resnet34's ImageNet weights: torchvision, BSD-3-Clause")
  await page.screenshot({ path: `${SHOTS}/size-balanced.png` })

  // Each size is a larger ResNet under the same Unet
  await size('Fast').click()
  await expect(params).toHaveText('14.3 M')
  await size('Accurate').click()
  await expect(params).toHaveText('32.5 M')
  await expect(page.getByTestId('time-here')).toHaveText(/^\d+(\.\d)? ms · CPU$/, { timeout: 120000 })

  // Advanced is folded away until opened; Custom opens it, and changing anything in it makes the size Custom
  const depth = page.getByLabel('Encoder depth')
  await expect(depth).toBeHidden()
  await size('Custom').click()
  await expect(depth).toBeVisible()
  await expect(size('Accurate')).toHaveAttribute('aria-checked', 'true')
  await depth.selectOption('4')
  await expect(size('Custom')).toHaveAttribute('aria-checked', 'true')
  await expect(params).not.toHaveText('32.5 M')
  await page.screenshot({ path: `${SHOTS}/size-custom.png` })

  // Only what trains is offered. FPN keeps its own depth, as its head upsamples by a fixed factor.
  // A select's label reads on into its options, so the field is found by how its label starts
  const encoder = page.getByLabel(/^Encoder(?! depth)/)
  await page.getByLabel('Architecture').selectOption('FPN')
  await expect(depth.locator('option')).toHaveText(['5 (its own)'])
  await expect(depth).toHaveValue('5')
  await expect(page.getByLabel('Attention')).toBeDisabled()
  // DeepLabV3 can't be built on DenseNet, or trained on one image a batch
  await page.getByLabel('Architecture').selectOption('DeepLabV3')
  // Playwright counts an option as enabled whatever its attribute says
  await expect(encoder.locator('option', { hasText: 'densenet121' })).toHaveAttribute('disabled', '')
  await expect(depth.locator('option')).toHaveText(['3', '4', '5 (its own)'])
  await page.getByLabel('Batch').fill('1')
  await expect(page.getByTestId('problem')).toHaveText('DeepLabV3 trains on batches of 2 or more.')
  await expect(page.getByRole('button', { name: /Train on 8 labelled/ })).toBeDisabled()
  await page.getByLabel('Batch').fill('2')
  await expect(params).toHaveText(/M$/, { timeout: 60000 })
  // An encoder an architecture can't take gives way to one it can
  await page.getByLabel('Architecture').selectOption('Unet')
  await encoder.selectOption('densenet121')
  await page.getByLabel('Input size').fill('64')
  await page.getByLabel('Architecture').selectOption('PAN')
  await expect(encoder).toHaveValue('resnet18')
  await expect(page.getByTestId('problem')).toHaveText('PAN needs an input size of 128 or more.')

  // A short run with every advanced setting changed trains here, and keeps them
  await size('Fast').click()
  await depth.selectOption('3')
  await page.getByLabel('Attention').selectOption('scse')
  // scSE squeezes a block's channels 16 to 1, more than half of Unet's narrowest block has
  await expect(page.getByLabel('Decoder width').locator('option', { hasText: '0.5×' })).toHaveAttribute('disabled', '')
  await page.getByLabel('Decoder width').selectOption('1.5')
  // The loss chart is labelled Loss too
  await page.getByRole('combobox', { name: 'Loss' }).selectOption('dice')
  await page.getByLabel('Learning-rate schedule').selectOption('cosine')
  await page.getByLabel('Warm-up epochs').fill('1')
  await page.getByLabel('Augmentation').selectOption('light')
  await page.getByLabel('Seed').fill('7')
  await page.getByLabel('Early-stopping patience, epochs').fill('5')
  await page.getByLabel('Epochs', { exact: true }).fill('2')
  await page.getByLabel('Input size').fill('64')
  await page.getByLabel('Learning rate').fill('0.001')
  await page.getByLabel('Start from ImageNet weights').uncheck()
  await expect(page.getByTestId('estimate')).toContainText('Unet · resnet18 · depth 3 · decoder 1.5× · scSE')
  await expect(page.getByTestId('time-here')).toHaveText(/^\d+(\.\d)? ms · CPU$/, { timeout: 120000 })
  await page.screenshot({ path: `${SHOTS}/size-advanced.png` })
  await page.getByRole('button', { name: /Train on 8 labelled/ }).click()
  // The old run shows Finished until the new one has started
  await expect(page.locator('.tile', { hasText: 'Status' })).toContainText('Training', { timeout: 120000 })
  await expect(page.locator('.tile', { hasText: 'Status' })).toContainText('Finished', { timeout: 300000 })
  const run = readdirSync(path.join(dir, '.myseg/runs')).find((n) => n !== OLD)!
  const runDir = path.join(dir, '.myseg/runs', run)
  const config = readFileSync(path.join(runDir, 'config.yaml'), 'utf8')
  for (const line of ['encoder_depth: 3', 'decoder_width: 1.5', 'decoder_attention_type: scse', 'seed: 7', 'type: cosine',
    'warmup_epochs: 1', 'strength: light', 'patience: 5', 'type: dice']) {
    expect(config).toContain(line)
  }
  const cost = readFileSync(path.join(runDir, 'events.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
    .find((e) => e.type === 'model')
  expect(cost.device).toBe('CPU')
  expect(cost.image_size).toBe(64)

  // Its export is what Seamtag 3.3 reads, as before
  await page.getByRole('button', { name: 'Export for Seamtag' }).click()
  await expect(page.getByTestId('export')).toContainText('model.onnx', { timeout: 120000 })
  const sidecar = JSON.parse(readFileSync(path.join(runDir, 'model.json'), 'utf8'))
  expect(Object.keys(sidecar).sort()).toEqual(['classes', 'input_size', 'mean', 'std', 'tile'])
  expect(sidecar.input_size).toEqual([64, 64])

  // Duplicate brings a run's settings back, to change one and train again
  await size('Accurate').click()
  await page.getByLabel('Epochs', { exact: true }).fill('30')
  await expect(page.getByTestId('run-network')).toHaveText('Unet · resnet18 · depth 3 · decoder 1.5× · scSE')
  await page.getByRole('button', { name: 'Duplicate' }).click()
  await expect(size('Custom')).toHaveAttribute('aria-checked', 'true')
  await expect(depth).toHaveValue('3')
  await expect(page.getByLabel('Decoder width')).toHaveValue('1.5')
  await expect(page.getByLabel('Attention')).toHaveValue('scse')
  await expect(page.getByLabel('Learning-rate schedule')).toHaveValue('cosine')
  await expect(page.getByLabel('Seed')).toHaveValue('7')
  await expect(page.getByLabel('Epochs', { exact: true })).toHaveValue('2')
  await expect(page.getByLabel('Input size')).toHaveValue('64')
  // A run from before the settings trained with what is now the recipe, so it comes back as its size
  await page.locator('.run', { hasText: '20260101 000000' }).click()
  await page.getByRole('button', { name: 'Duplicate' }).click()
  await expect(size('Fast')).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByLabel('Input size')).toHaveValue('512')
  await expect(page.getByLabel('Start from ImageNet weights')).toBeChecked()

  // Runs side by side: the old one has no costs, the new one its own
  await page.getByRole('button', { name: 'Compare runs' }).click()
  const rows = page.getByTestId('compare').locator('tbody tr')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText('Unet · resnet18 · depth 3 · decoder 1.5× · scSE')
  await expect(rows.nth(0)).toContainText(/64 px.*\d+\.\d M.*ms\s*CPU/)
  await expect(rows.nth(1)).toHaveText(/2026-01-01 00:00\s*Unet · resnet18\s*512 px\s*–\s*–\s*0\.500/)
  await page.screenshot({ path: `${SHOTS}/size-compare.png` })
  await rows.nth(0).click()
  await expect(page.getByTestId('run-network')).toHaveText('Unet · resnet18 · depth 3 · decoder 1.5× · scSE')

  // On the GPU machine the memory is with mixed precision, and a tile is timed there only when asked
  await page.getByLabel('Run on').selectOption('ssh')
  await page.getByLabel('SSH host').fill('gpu@fake')
  await page.getByLabel('Remote folder').fill('seamline')
  await expect(page.getByTestId('estimate')).toContainText(/GPU memory at batch 8\s*≈ \d+\.\d GB/)
  await expect(page.getByTestId('time-there')).toHaveText('–')
  expect(gpu.estimates()).toEqual([])
  await page.getByRole('button', { name: 'Time a tile on gpu@fake' }).click()
  await expect(page.getByTestId('time-there')).toHaveText(/^\d+(\.\d)? ms · CPU$/, { timeout: 180000 })
  expect(gpu.estimates()).toHaveLength(1)
  expect(gpu.estimates()[0]).toContain('"image_size":512')
  await page.screenshot({ path: `${SHOTS}/size-remote.png` })

  expect(errors).toEqual([])
  await app.close()
})
