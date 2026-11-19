# Seamtag's label files

Fixtures for the engine's tests: Seamline must train on every shape and label format Seamtag
saves, and read them as Seamtag and labelme read them.

## Shapes and their masks

`shapes/` holds label files written by labelme 7.9.1's own code, one for each kind of shape,
each beside its 64x48 image. `masks/` holds the index masks labelme's `shapes_to_label` makes of
them, each pixel the number `masks/classes.json` gives its class. `make_masks.py` writes both;
from this folder:

```
uv run --no-project --python 3.12 --with labelme==7.9.1 --with pillow==13.0.0 python make_masks.py
```

`tests/unit/test_studio_shapes.py` compares Seamline's masks with these.

## LabelImg's formats

`labelimg/` is a folder as a project has one, with Pascal VOC, YOLO and CreateML files:

- `voc/`, `yolo/`, `yolo-hand/` and `createml/` hold LabelImg 1.8.6's own files, copied from
  Seamtag's `tests/fixtures/labelimg/1.8.6`, where its `make_labelimg_files.py` wrote them with
  LabelImg's code. `createml-master/` holds LabelImg master's CreateML file, which keeps
  `verified`. Their images are plain PNGs of the sizes the files give, as Seamtag's fixtures have
  none.
- `other/` holds other tools' files, and files that aren't labels: a caption beside no
  `classes.txt`, a CreateML list of another image, a Pascal VOC file written by hand, and an
  image with both LabelMe and Pascal VOC files.
- `bom/` and `bom-yolo/` start their files with a byte order mark.
- `odd/` holds YOLO numbers as JavaScript reads them, and blank lines in `classes.txt`.
- `broken/` holds files neither app can read.

`labelimg.json` is what Seamtag 3.3 finds and reads for each image there, made by running
Seamtag's own `findLabels` and `readLabels` with `read_with_seamtag.mjs`. From a Seamtag
checkout, with its Node:

```
node_modules/.bin/esbuild <Seamline>/tests/fixtures/seamtag/read_with_seamtag.mjs --bundle \
  --platform=node --format=esm --alias:seamtag=. --outfile=/tmp/read_with_seamtag.mjs
node /tmp/read_with_seamtag.mjs <Seamline>/tests/fixtures/seamtag/labelimg \
  > <Seamline>/tests/fixtures/seamtag/labelimg.json
```

`tests/unit/test_studio_formats.py` compares Seamline's readers with it, file by file. A byte
order mark at the start of `classes.txt` (`bom-yolo/`) is no part of the first class's name in
either app, where LabelImg and Seamtag 3.2.0 keep it.

## LabelMe checks

`labelme_checks.json` lists LabelMe files, each as its `text`, with the `error` Seamtag's
`parseLabelFile` (3.2.0, dfe262e) gives for it, or `null` when it reads. They cover every check
both apps make, and which fault is named when a file has more than one. Where the error is
`It isn't valid JSON`, the JSON parser's own words follow it in brackets, and those differ between
JavaScript and Python. The engine's `tests/unit/test_studio_formats.py` and the app's
`app/e2e/shared.spec.ts` both run every case, and so does Seamtag's `tests/unit/labelfile.test.ts`
on its copy in `tests/fixtures/seamline/`, so the engine, the app and Seamtag accept and refuse the
same files in the same words. Change both copies together.
