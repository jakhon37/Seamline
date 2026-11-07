<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/seamline-logo-on-dark.svg">
    <img src="docs/brand/seamline-logo-on-light.svg" alt="Seamline" width="340">
  </picture>
</p>

<p align="center"><b>Label images and train segmentation models in one desktop app, on your own machines.</b></p>

<p align="center">
  <a href="https://github.com/jakhon37/Seamline/actions/workflows/tests.yml"><img src="https://github.com/jakhon37/Seamline/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/jakhon37/Seamline/actions/workflows/lint.yml"><img src="https://github.com/jakhon37/Seamline/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

![Seamline reviewing a model's proposals on a circuit board: a melted solder joint and a missing sensor](docs/media/hero.webp)

Label some images, train, let the model label the rest, review what it proposed, least certain first, and train again on what you accepted. It all stays on your own machines: labels are plain LabelMe files beside your images, and training runs on this computer or on your GPU machine over SSH.

- **Review ordered by uncertainty.** Every proposed shape gets a confidence score, and you can work through the least certain first, accepting, rejecting or skipping by keyboard.
- **Detached remote training.** A run on your GPU machine keeps going if you close the app or lose the connection, and the app picks it back up.
- **Tile pyramids for large images.** The viewer draws only the tiles on screen, from a pyramid built once with libvips.
- **mySegLib underneath.** Training goes through the library in this repo, built on PyTorch and segmentation_models_pytorch. It also works on its own, with 9 architectures, multi-GPU DDP, chunked inference and ONNX export.

## How it works

<picture>
  <source media="(max-width: 700px)" srcset="docs/media/workflow-phone.webp">
  <img src="docs/media/workflow.webp" alt="The workflow: label, train, pre-label, review, then train again on the accepted shapes">
</picture>

## See it work

![A 32-second demo: reviewing proposals by keyboard, rejecting a wrong one, touching one up with the brush, and comparing two training runs](docs/media/demo.gif)

This is the real app on real images, driven by a script. The cursor and key labels are drawn on top for the recording. There's also an [MP4 version](docs/media/demo.mp4).

<table>
  <tr>
    <td width="50%"><img src="docs/media/label.webp" alt="Hand-drawn polygons on two missing sensors"></td>
    <td width="50%"><img src="docs/media/review.webp" alt="A proposed melted solder shape, with the model's certainty and the review keys"></td>
  </tr>
  <tr>
    <td><b>Label</b> with polygons, rectangles, a brush and an eraser</td>
    <td><b>Review</b> each proposal: <code>W</code> accept, <code>S</code> reject, <code>Tab</code> skip</td>
  </tr>
  <tr>
    <td><img src="docs/media/brush.webp" alt="The brush joining two proposed pieces of one melted pin"></td>
    <td><img src="docs/media/train.webp" alt="Training curves and log from a real GPU run, with two runs listed"></td>
  </tr>
  <tr>
    <td><b>Touch up</b> with a brush, here joining two pieces of one melted pin</td>
    <td><b>Train</b> on a GPU machine over SSH, and compare runs</td>
  </tr>
</table>

## How it's built

![Architecture: the window, main process and Python engine on your computer, with detached training on a GPU machine over SSH](docs/media/architecture.webp)

Seamline is an Electron app: a React window, a Node main process, and a Python engine that reports in JSON lines and trains through mySegLib.

- **Scores rank the proposals.** Pre-labelling runs the newest model on every image with no accepted shapes and saves as it goes. A shape's score is [the model's mean confidence inside it](engine/studio.py), which is good for ranking but isn't a probability. Each image's shapes are reviewed least certain first, and the **Least certain first** order sorts the images the same way. Proposals stay out of training until you accept or edit them.
- **Remote runs outlive the app.** [The app](app/src/main/index.ts) checks the machine's GPU and disk, uploads the dataset with rsync and starts training with `setsid nohup`. The run writes its progress to a file the app reads every 3 seconds, so closing the app, opening another folder or losing the connection doesn't touch it. When the project opens, the app reattaches. When the run ends, the best model comes back and the run's files are removed from the machine. [A stand-in for the GPU machine](app/e2e/fake_gpu.ts) tests the failure cases without needing a GPU, and the normal path has run on a real GTX 1080 Ti.
- **Large images open fast.** Large images and TIFFs get a pyramid built once with libvips and reused until the file changes, and the viewer draws only the tiles on screen. On an older PC, a 6050×12300 test TIFF took 8 seconds to build its pyramid, and after that the open call returned in 4 ms.
- **Training goes through mySegLib.** The engine turns LabelMe JSON into masks and tiles, then trains through the library's `scripts/train.py`. The library also works on its own from YAML configs and the command line: see the [library guide](docs/LIBRARY.md).
- **Labels are hard to lose.** Each save names its own folder and goes through one ordered queue, the last edit is flushed before the window closes, and a label file that can't be read is flagged and never written over.
- **Geometry checked against an independent model.** The brush and eraser merge and cut polygons with Clipper. [A seeded test](app/e2e/geometry.spec.ts) runs 150 random paint and erase strokes and checks over 10,000 points against an independent model.

## Tested on real images

The screenshots, the demo and these numbers come from real runs on circuit-board photos from the [VisA dataset](https://github.com/amazon-science/spot-diff): four defect types, 64 labelled images, and 27 held back that the model never saw. The model is a Unet with a ResNet-34 encoder, trained through the app on a GTX 1080 Ti. It's one run on a small set, so read the numbers as rough. The scripts and results, and what these runs changed in the training code, are in [`docs/demo`](docs/demo/README.md).

On the 27 unseen images, all with defects, the model found 24 of the 55 defects (at least half of each covered), proposed nothing on 6 images, and reached a mean IoU of 0.41 across the four defect types. That IoU is over all pixels of each type. The app's Best IoU of 0.669 reads higher because it's on validation tiles and averages background in.

**Mostly right when it proposes.** Of the 31 proposals, 24 are on a defect and the right type, 3 are on a defect but the wrong type, and 4 are mostly off any defect. On 30 defect-free boards, it proposed only 1 shape.

![The same unseen images with true labels and the model's proposals](docs/media/examples.webp)

These examples are hand-picked to show each defect type, so they're better than average for all but bent pins. The averages are in the chart:

![Bar charts: IoU by defect type, and IoU by certainty quarter](docs/media/results.webp)

The right-hand chart is the case for reviewing least certain first: low scores go with weak shapes (Spearman 0.73 between score and IoU). It's from only 31 shapes, so treat it as a direction rather than a measurement.

**What it still gets wrong.** With 64 labelled images, the model misses more than half of the defects and never finds a bent pin. A defect the model misses has no shape to review, so you still need to look at the images. How much labelling time this saves hasn't been measured.

## Try it

You need Node 20+ and Python 3.10+. There's no installer yet, and it's only been tested on Linux.

```bash
git clone https://github.com/jakhon37/Seamline.git
cd mySegLib
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd app
npm install
npm run build
npm start
```

Open any folder of images. [`app/README.md`](app/README.md) has the keyboard shortcuts and how to train on a GPU machine.

## Still to do

- An installer, and testing on Windows and macOS
- A way to mark an image as truly empty: rejecting every shape sends it back to be pre-labelled again
- Training on clean images too: tiles with no defect are skipped today
- Review of the model's worst errors, not just its least certain shapes
- SAM-assisted labelling

## License and credits

MIT, see [LICENSE](LICENSE). Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Demo images are from [VisA](https://github.com/amazon-science/spot-diff) (Zou et al., ECCV 2022), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and shown cropped with labels drawn on them.
