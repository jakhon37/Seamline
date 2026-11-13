# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Seamline, a desktop app for labelling and training, in `app/` and `engine/`
- `--events` option in `scripts/train.py` for per-epoch progress
- Images in subfolders, named by their path in the folder, such as `line2/img_007.jpg`, and JFIF,
  GIF, WebP and AVIF images, as Seamtag lists them; label files Seamline writes in a subfolder give
  `imagePath` from their own folder, as labelme reads it, and images deep in folders with long
  names train too
- Pascal VOC (`name.xml`), YOLO (`name.txt` beside a `classes.txt`) and CreateML (a list in
  `name.json`) labels, found and read as Seamtag finds and reads them, so Seamline trains on every
  label format Seamtag saves; it never writes or converts these files, so their images open
  read-only and pre-labelling skips them; a byte order mark at the start of `classes.txt` isn't
  part of the first class's name, where LabelImg and Seamtag 3.2.0 keep it, and a YOLO file
  Seamtag changes while the dataset is prepared trains as it was first read
- labelme 7's other shapes in the editor, shown as labelme shows them: circles, rotated boxes,
  masks, lines, line strips and points
- A Label in Seamtag button, which saves what's pending and opens Seamtag on the image, with the
  whole folder; for an image in a subfolder that needs Seamtag 3.3, as Seamtag 3.2.0 opens just the
  subfolder; while Seamtag starts, the button says so and can't be pressed again; a copy that
  closes as it starts gives way to the next one found, and one you picked that no longer starts is
  forgotten once another does; when it can't find Seamtag, or none starts, it says so and offers
  the download page or lets you pick the program, which it remembers once it has started, and
  `SEAMLINE_SEAMTAG` names the program to use
- An Export for Seamtag button on the Train page, and an `export` command in `engine/studio.py`,
  that write a run's model as `model.onnx` in its folder, beside the `model.json` Seamtag's
  pre-labelling reads; with onnxruntime installed, an export is saved only if its scores match
  PyTorch's to within 0.001, or 0.1% of the largest score when that's more; PSPNet runs export
  at any input size and with EfficientNet encoders, and a failed export says why
- Reading the folder again whenever Seamline's window or its Label page comes back into view, so
  images, labels and classes another app added, removed or changed show up, and a label file that
  couldn't be read is tried again; saves don't wait for it
- In the dataset check, the label formats it read and the shapes it left out for having no area,
  and an Add as classes button for labels that aren't in the class list
- Beside the Train button, the labels that aren't classes yet and so don't train, with how many
  images and shapes use them, and an Add as classes button; the shape list marks each shape whose
  label isn't a class

### Changed
- Images and masks pair by exact name, so one missing file can't shift every later pair. A mask
  with the image's own name wins over one that matches only without `_mask`, so `a.png` and
  `a_mask.png`, or `x/a.png` and `x_mask/a.png`, each train on their own mask
- Saves and pre-labelling write nothing over a label file another app changed since Seamline read
  it, such as Seamtag saving the same image, even when the change kept the file's size and time; a
  save that would is dropped, and the window reloads the labels and says so
- A label file Seamtag can't read either, such as broken XML, a YOLO line whose class isn't in
  `classes.txt` or a shape of an unknown type, stops the dataset check, which names the file by its
  path in the folder, and is marked red in the image list; unknown shape types with three or more
  points used to train as polygons, and shapes without a label were skipped
- LabelMe files are checked as Seamtag checks them, in its words, by the app and the engine alike:
  a file without a shapes list, or with a shape whose `group_id` isn't a whole number, whose
  `description` isn't text or whose `locked` isn't true or false, stops the dataset check too; a
  file without a shapes list used to count as not labelled
- The dataset check's list of labels not in the class list names only labels of shapes with an
  area, which would train once added
- Image files whose names start with a dot are left out, as Seamtag leaves them out
- An image counts as labelled, in the top bar, the Label and Done filters and the Train button,
  once it has an accepted shape with an area whose label is a class, as in training, and the counts
  follow the class list as it changes; pre-labelling still leaves alone any image with an accepted
  shape
- A folder opened without a class list starts with the labels of shapes with an area, leaving out
  labels used only on lines, points and AI proposals
- Pre-labelling leaves alone an image whose LabelMe file has no shapes, as when every shape was
  rejected, as Seamtag does, even if the file was emptied after pre-labelling started; such images
  used to be pre-labelled again. Where the model finds nothing, in a batch or on one image, the
  image keeps its earlier proposals rather than being left with an empty label file
- Undoing or deleting every shape on an image that had no label file, as after drawing a shape and
  taking it back, or rejecting what Pre-label proposed on that image, leaves it with no label file
  rather than an empty one, which reads as labels someone emptied: such an image stays out of
  pre-labelling in both apps, and Seamtag counts it as done. Seamline removes only a file it made
  since it started that no other app has changed since; a file Pre-label N images wrote stays,
  emptied
- `requirements.txt` asks for Pillow 12 or later: older Pillow opens 16-bit grey PNGs as 32-bit
  numbers, which train as white, and can't open AVIF images
- Saving the class list merges it with `.myseg/project.json` as it is then, which Seamtag writes
  too: classes another app added, recoloured or deleted there since Seamline read it stay that way,
  unless Seamline changed the same class, and the file's other keys are kept

### Fixed
- Training applied softmax twice: `activation` from the config reached the model, and the losses
  apply softmax themselves. A perfect prediction still cost 0.30, and small classes could collapse
  into background. `activation` now applies at inference and export only. Saved models still load,
  but retrain any model trained before this fix.
- Tiling dropped any leftover strip narrower than half a tile, so 30% of a 1404×1070 photo never
  reached training, and 53% of a 1500×1500 image. Tiles now cover each side fully.
- Local training could die with exit code -4 on CPUs with AVX but no AVX2: a race in MKL's first
  vector-math call. One single-threaded call at import took real runs from 6 of 12 crashing to 0 of
  30, and a stress loop to 0 of 6000. `MKL_ENABLE_INSTRUCTIONS=AVX` only hides it: the same loop
  still crashed 578 times.
- Line strips and point sets with three or more points were filled as polygons in training. Lines,
  line strips and points have no area, so training now leaves them out, and the dataset check
  counts them.
- Circles and labelme's masks drew nothing in training masks, so an image whose only shapes were
  circles, masks, lines or points could train as all background. Circles and masks now cover the
  pixels labelme 7.9.1 gives them, and the dataset check counts an image with no shape that has an
  area as not labelled.
- Predictions took their class names from the project's class list as it is now, so once classes
  were added, removed or reordered, a run's proposals could get the wrong names. Runs now keep
  their classes in `classes.json`; runs trained before this still take the project's.
- The editor could change shapes it doesn't draw: the brush and eraser turned rotated boxes, line
  strips and point sets they touched into polygons, and circles, masks and locked shapes could be
  dragged. Its tools now move and reshape only polygons and rectangles that aren't locked, and
  saving keeps every other shape as it was. Circles and masks were also drawn as lines.
- BMP images were listed but couldn't be opened, since the libvips in sharp doesn't read them. The
  window now shows them itself.
- Saving a LabelMe file dropped the image labelme keeps in it (`imageData`), and rewrote whole
  numbers past 2^53, such as ids, with other digits. Both now stay as they were.
- Two images of the same size and time whose names differ only in characters other than A-Z, a-z,
  0-9, `_`, `.` and `-`, such as `a b.tif` and `a_b.tif`, shared one tile pyramid, so one of them
  showed the other's pixels.
- The engine read label files and `.myseg/project.json` in the system's default text encoding
  rather than as UTF-8, so labels outside ASCII could be misread where that default isn't UTF-8,
  as on Windows.
- An image whose only shapes have labels that aren't in the class list, such as labels typed in
  Seamtag, trained as all background. It now counts as not labelled until those labels are added
  as classes.
- A run that stopped before training, such as on a label file that can't be read, lost its reason
  once its live log was gone, and the run list showed it as stopped. The run now keeps the reason
  and shows as failed.
- 16-bit greyscale images trained, and were pre-labelled, as almost all white, since each pixel was
  cut off at 255. Each pixel now keeps its high byte, as Seamtag reads it.
- Uncompressed greyscale TIFFs that their EXIF orientation turns (5 to 8) trained scrambled, at
  their stored size, so their labels fell on the wrong pixels. They now train upright, as both apps
  show them.
- Training read images without their ICC profile, while the window and Seamtag show them through
  it, so photos in Adobe RGB, Display P3 or CMYK trained, and were pre-labelled, on other colours
  than the ones shown. Images are now read through their profile, as libvips reads them; CMYK
  images that have no profile still differ.
- A run on a GPU machine was named by UTC time, and a run on this computer by local time, so away
  from UTC a remote run could sort below a local one started hours before it, and pre-labelling went
  on with the older model. Remote runs are now named by local time too, a second later when a run
  already has that name.
- Once Pre-label N images had finished, its button could stay as Stop, with Pre-label greyed out,
  until another folder was opened or Seamline restarted: the batch's last progress can reach the
  window after its end, and no longer shows on the button then.

## [0.1.0] - 2026-11-05

First public release.

### Added
- YAML-driven training, evaluation and inference scripts
- Models from segmentation_models_pytorch, with focal, Dice, cross-entropy and combined losses
- Multi-GPU training with DDP, and a DataParallel fallback
- Mixed precision, gradient accumulation, gradient checkpointing and early stopping
- Chunked inference for large images, and ONNX export
- Unit tests, GitHub Actions workflows, pre-commit hooks and Docker support
