# Seamline

Label images and train segmentation models in one desktop app, built on [mySegLib](../README.md). Press `Ctrl+1` for Label and `Ctrl+2` for Train; each keeps its place when you switch.

## Run

Needs Node 20+ and the repo's Python requirements in `.venv` at the repo root. Set `MYSEG_PYTHON` to use another interpreter that has them: with Pillow older than 12, 16-bit grey PNGs train as white and AVIF images can't be read.

```bash
npm install
npm run build
npm start
```

Open any folder of images. Seamline lists the JPEG, JFIF, PNG, BMP, GIF, TIFF, WebP and AVIF images in it and its subfolders, leaving out files and folders whose names start with a dot, such as `.myseg`, and folders linked from elsewhere. BMP images are shown whole rather than from a tile pyramid, since the libvips in sharp can't read them. Labels are saved beside each image as LabelMe JSON, so labelme and [Seamtag](https://github.com/jakhon37/Seamtag) open the same folder. Saving keeps the rest of the file as it was, including the image labelme keeps in it. A label file that can't be read is marked red and never written over. Project settings, the tile cache, the prepared dataset and runs go in `.myseg/` inside that folder.

## Keys

| Key | Action |
|---|---|
| `V` / `P` / `R` | Select and edit / polygon / rectangle |
| `B` / `E`, `[` `]` | Brush / eraser, brush size |
| `W` / `S` / `Tab` | Accept / reject / skip the AI shape, then go to the next |
| `Shift+W` | Accept every AI shape here and go to the next image to review |
| Click first point, `Enter` or double-click | Close polygon |
| `1`-`9` | Choose class, or change the selected shape's class |
| `A` / `D` | Previous / next image |
| Wheel, `Space`+drag or right-drag, `0` | Zoom, pan, fit |
| `F` | Show or hide shapes |
| Double-click an edge / `Alt`+click a point | Add / remove a point |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |

## How it fits together

- `src/main`: file access, finding and reading label files, image tile pyramids (libvips through sharp), and starting the Python engine and Seamtag
- `src/shared`: what the main process and the window both use, such as the Pascal VOC, YOLO and CreateML readers in `formats.ts`, ported from LabelImg
- `src/renderer`: the React UI, with `Labeler.tsx` and `Trainer.tsx` side by side
- `../engine/studio.py`: `prepare` turns label files into masks and tiles, `train` trains through `scripts/train.py`, `predict` pre-labels, and `export` writes a run's model as ONNX for Seamtag

Pre-labelled shapes stay dashed and are left out of training until you accept or edit them.

## Review pre-labels

**Pre-label N images** runs the newest model on every image with no label file, or a LabelMe file with only AI shapes, saving as it goes. As in Seamtag, a label file with no shapes, as when every shape was rejected, is left alone, even if it was emptied after pre-labelling started, and so are labels another app changed since Seamline last read them. An image that had no label file has none again once you undo or delete every shape on it, as after drawing a shape and taking it back, or rejecting what **Pre-label** proposed on that image, so it stays to label and to pre-label in both apps. Seamline removes only a file it made for that image since it started, and only if no other app has changed it since. A file that **Pre-label N images** wrote is kept: rejecting every shape in it leaves it empty. Where the model finds nothing, here or with **Pre-label** on one image, nothing is written, so earlier proposals stay. Press `W` to step through the shapes. **Least certain first** orders images by their least certain shape, and each image's shapes are reviewed in the same order. A shape's score is the model's mean confidence inside it. It's good for ranking but isn't a probability, and an object the model missed has no shape, so no score. `Ctrl+Z` right after moving on goes back to the last image and takes the decision back.

## Brush and eraser

The brush paints the selected shape's class, or the current class if nothing is selected. Shapes of that class it touches merge into one and keep their layer, which also joins shapes split at a tile edge. Where another class sits on a higher layer, it stays on top: erase it to paint there. The eraser cuts from the selected shape, or from everything it touches if nothing is selected, and can split a shape in two. Shapes Seamline doesn't reshape, such as circles, rotated boxes, masks and locked shapes, are left as they are.

LabelMe polygons can't have holes, so a hole is saved as one outline with a zero-width cut. The cut shows as a thin line on screen, but it doesn't change the training mask.

## Labels from labelme, LabelImg and Seamtag

Seamline finds an image's labels beside it, in the order Seamtag looks by default: `name.json` (LabelMe JSON, or a CreateML list), `name.xml` (Pascal VOC), then `name.txt` with a `classes.txt` in the same folder (YOLO). Names match as the system matches them: whatever their case on Windows and macOS, and whatever their Unicode form on macOS. Other `.txt` files, such as captions, aren't labels, and Seamline doesn't read labels Seamtag saves in another folder, with `-O` or a changed output folder. Labels in every format count in the image list, its filters and the Train page. As in training, an image is labelled once it has an accepted shape with an area whose label is a class: lines and points alone don't make it labelled, though pre-labelling still leaves such an image be. An image whose only shapes with an area have labels that aren't classes, such as labels typed in Seamtag, doesn't train until those labels are added as classes. The Train page says so beside the Train button, with **Add as classes**, and the shape list marks such shapes *not a class*. A LabelMe file the engine and Seamtag can't read, such as one with a shape that has no label or a type labelme doesn't have, is marked red like any other file that can't be read, and is read again each time Seamline reads the folder.

Pascal VOC, YOLO and CreateML keep boxes. Seamline trains on them but never writes or converts them, so their images are marked VOC, YOLO or CreateML in the list and open read-only, with a bar that says so and a **Label in Seamtag** button.

labelme 7's other shapes show as labelme shows them: circles, rotated boxes, masks in their class colour, lines, line strips and points. Seamline's tools move and reshape only polygons and rectangles that aren't locked. Other shapes, and locked ones, can be selected, relabelled, accepted or rejected, and deleted, and are otherwise saved exactly as they were; the shape list names each one's type. Lines, line strips and points have no area, so training leaves them out, and the dataset check on the Train page counts them.

## Working with Seamtag

**Label in Seamtag**, in the top bar, saves what's waiting to be saved and opens [Seamtag](https://github.com/jakhon37/Seamtag) on the image you're on, with the whole folder. For an image in a subfolder, that needs Seamtag 3.3 or later: Seamtag 3.2.0 opens just the subfolder. Seamline looks for Seamtag where you last picked it, on the `PATH`, and where its installers put it on Linux, macOS and Windows; set `SEAMLINE_SEAMTAG` to name the program instead. A copy that closes as it starts gives way to the next one found, and a copy you picked that no longer starts is forgotten once another one does. If it can't find Seamtag, or no copy starts, Seamline says so and offers the download page, or lets you pick the program, which it remembers once it has started. On Ubuntu 24.04 and later, install Seamtag's `.deb`: its AppImage needs `--no-sandbox` there, which Seamline doesn't add.

Both apps can have the folder open at once. Whenever you come back to Seamline's window, or to its Label page, it reads the folder again: images and labels another app added, removed or changed show up, including on the image you're on, which stays chosen. Seamline never writes over labels another app changed since it read them, even to the same size and time: a save that would is dropped, and Seamline reloads the labels and says your last change wasn't saved. Pre-labelling skips such labels too. Seamtag 3.3 and later do the same for Seamline's changes. Labels new to the folder don't become classes by themselves: the Train page names those that would train, beside the Train button and in the dataset check, and **Add as classes** adds them.

Both apps keep the class list in `.myseg/project.json`. When Seamline saves it, classes another app added, recoloured or deleted there since Seamline read it stay that way, unless Seamline changed the same class, and other keys in the file are kept. Coming back to Seamline shows the list as it is then. A folder opened without a class list starts with the labels of its accepted shapes that have an area.

On the Train page, a run with a saved model has **Export for Seamtag** once it has stopped. It writes the model as `model.onnx` in the run's folder, `.myseg/runs/<run>/`, beside a `model.json` that gives the run's classes, input size, tile and normalisation. Seamtag 3.3 and later list every run exported so under AI → Model for pre-labelling, newest first, and pre-label with the newest unless a model was chosen for the folder there or with AI → Choose a model…. In Seamtag 3.2.0, pick the model with AI → Choose a model…, and show hidden files in its file dialog to see `.myseg` (Ctrl+H, or Cmd+Shift+. on a Mac). **Open in Seamtag** opens the folder there. A run trained before runs kept their class names exports with the project's classes, as long as there are as many as the model learned. From the repo root, `.venv/bin/python engine/studio.py export <folder> --run <run>` does the same.

## Train on a GPU machine

On the Train page, set **Run on** to **GPU machine over SSH**, then enter `user@host` and a remote folder. You need key-based SSH, rsync on both ends, and a GPU PyTorch for the machine's `python3`.

The app prepares the dataset here, checks there is 3 GB to spare, and uploads it. Training runs detached, so it survives closing the app, opening another folder or a lost connection; the app reattaches when the project opens. When the run ends, the best model comes back and the run's files are removed from the machine. A run that ends while its folder isn't open waits on the machine until you open that folder again. Another folder can't be opened until a run has finished starting.

A run that dies before it reports anything, or whose files are removed from the machine, ends as Failed with a message. A run that died keeps its output in `out.log` in the run's folder.

`engine/remote_setup.py` builds an isolated environment in the remote folder, linking the machine's PyTorch rather than copying it.

## Test

```bash
npm run test:e2e
```

`e2e/remote-switch.spec.ts` needs no GPU: `e2e/fake_gpu.ts` stands in for the machine. Set `MYSEG_E2E_SSH=user@host` to include the test on a real one.

`e2e/geometry.spec.ts` and `e2e/shared.spec.ts` test plain functions, without the app: the brush and eraser, the class list merge, the labelled count, and the LabelMe file checks, on the same cases as the engine's tests (`../tests/fixtures/seamtag/labelme_checks.json`).

`e2e/formats.spec.ts`, `e2e/two-apps.spec.ts` and `e2e/seamtag.spec.ts` work on folders as Seamtag leaves them, with label files from Seamtag's own tests. Each launch gets a home folder of its own, and `SEAMLINE_SEAMTAG` names a stand-in for Seamtag, or nothing, so they never touch your settings or start the Seamtag installed on the machine. Where they need the engine, a few lines of Node stand in for it. The tests open the app's window: run them on a virtual X server, such as Xvfb, to keep it off your screen.

The engine's tests are in `../tests/unit/test_studio_*.py`, and [`../tests/fixtures/seamtag`](../tests/fixtures/seamtag/README.md) holds label files written by labelme 7.9.1 and LabelImg, with the masks and readings they're compared with, and says how they were made.
