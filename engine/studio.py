#!/usr/bin/env python3
"""Backend commands for the desktop app. Each prints JSON or writes JSONL events.

prepare: project folder with label files -> train/val images and index masks
train:   prepare, write a run config and call scripts/train.py
predict: run a trained model on images and print its polygons as LabelMe shapes
export:  write a trained run as ONNX, with the sidecar Seamtag's pre-labelling reads
"""
import argparse
import base64
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps, TiffImagePlugin, features

try:
    from PIL import ImageCms
except ImportError:  # a Pillow built without Little CMS: colours as stored
    ImageCms = None

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
Image.MAX_IMAGE_PIXELS = None
# Pillow's own reader takes an uncompressed one-band TIFF that EXIF turns (orientation 5 to 8) at
# the wrong width, so it comes out scrambled. Through libtiff, which reads compressed TIFFs, it's
# right.
TiffImagePlugin.READ_LIBTIFF = features.check("libtiff")
# What Seamtag lists, so both apps see the same images
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".avif"}
# labelme 7's shape types. Lines and points have no area, so they don't train.
AREA_TYPES = ("polygon", "rectangle", "oriented_rectangle", "circle", "mask")
SHAPE_TYPES = (*AREA_TYPES, "line", "linestrip", "point", "points")
LABEL_FORMATS = ("labelme", "voc", "yolo", "createml")
YOLO_CLASSES = "classes.txt"
# Tile names are this, then __left_top.png, which keeps them well inside 255 bytes
TILE_STEM_BYTES = 150


def load_project(root: Path) -> dict:
    return json.loads((root / ".myseg" / "project.json").read_text(encoding="utf-8"))


def list_images(root: Path) -> list:
    """Images in the folder and its subfolders, as Seamtag lists them: hidden files and folders,
    such as .myseg, are left out, and folders behind links aren't followed."""
    found, folders = [], [root]
    while folders:
        try:
            entries = list(os.scandir(folders.pop()))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                folders.append(Path(entry.path))
            elif Path(entry.name).suffix.lower() in IMAGE_EXTS:
                found.append(Path(entry.path))
    return sorted(found)


# Label files, found and read as Seamtag finds and reads them, so both apps see the same labels


def read_text(path: Path) -> str:
    # As Seamtag reads label files: UTF-8, a BOM kept, bytes that aren't UTF-8 replaced
    return path.read_bytes().decode("utf-8", errors="replace")


def parse_json(text: str):
    """JSON as Seamtag parses it, where NaN and Infinity aren't numbers"""

    def refuse(name):
        raise ValueError(f"{name} isn't a number")

    try:
        return json.loads(text, parse_constant=refuse)
    except ValueError as error:
        raise ValueError(f"It isn't valid JSON ({error})") from None


def find_labels(path: Path):
    """The image's label file, where Seamtag looks and in its order: <stem>.json (LabelMe, or a
    CreateML list), <stem>.xml (Pascal VOC), then <stem>.txt beside a classes.txt (YOLO). Other
    .txt files, such as captions, aren't labels. None when it has none."""
    for suffix in (".json", ".xml"):
        if path.with_suffix(suffix).exists():
            return path.with_suffix(suffix)
    txt = path.with_suffix(".txt")
    if txt.name != YOLO_CLASSES and txt.exists() and (txt.parent / YOLO_CLASSES).exists():
        return txt
    return None


def read_label_file(file: Path, image: Path, size: tuple = (1, 1), texts: dict = None) -> tuple:
    """The label file's format and its shapes, as LabelMe shapes. YOLO's boxes are shares of the
    image, so they need its upright size. Raises ValueError for a file Seamtag can't read either.
    Files read are kept in texts, when given, and read from there again, so a file read twice at
    two sizes is the same file both times."""

    def read(path):
        if texts is None:
            return read_text(path)
        if path not in texts:
            texts[path] = read_text(path)
        return texts[path]

    text = read(file)
    if file.suffix == ".xml":
        fmt, shapes = "voc", read_voc(text)
    elif file.suffix == ".txt":
        classes = read(file.parent / YOLO_CLASSES)
        known = read_classes(classes) if classes.strip(JS_SPACE) else []
        fmt, shapes = "yolo", read_yolo(text, known, *size)
    elif looks_like_createml(text):
        # Entries name images by file name. A list that doesn't name this one has no labels of it.
        fmt, shapes = "createml", read_createml(text, image.name) or []
    else:
        return "labelme", read_labelme(text)
    for index, shape in enumerate(shapes):
        check_shape(shape, index)
    return fmt, shapes


def is_flags(value) -> bool:
    return isinstance(value, dict) and all(isinstance(v, bool) for v in value.values())


def read_labelme(text: str) -> list:
    """A LabelMe file's shapes, checked as Seamtag's parseLabelFile checks the file, in its order"""
    doc = parse_json(text)
    if not isinstance(doc, dict):
        raise ValueError("It isn't a LabelMe file: the top level must be an object")
    shapes = doc.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("It isn't a LabelMe file: there is no shapes list")
    if doc.get("imagePath") is not None and not isinstance(doc["imagePath"], str):
        raise ValueError("imagePath must be text")
    if doc.get("imageData") is not None and not isinstance(doc["imageData"], str):
        raise ValueError("imageData must be base64 text")
    if doc.get("flags") is not None and not is_flags(doc["flags"]):
        raise ValueError("Image flags must map names to true or false")
    for index, shape in enumerate(shapes):
        check_shape(shape, index)
    for key in ("imageHeight", "imageWidth"):
        value = doc.get(key)
        if value is not None and not (is_whole(value) and value > 0):
            raise ValueError(f"{key} must be a positive whole number")
    return shapes


def is_number(value) -> bool:
    # As JavaScript reads JSON: true and false aren't numbers, and none is past a double's range
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) <= sys.float_info.max
    )


def is_whole(value) -> bool:
    # JavaScript's Number.isInteger, which takes 5.0 as well as 5
    return is_number(value) and float(value).is_integer()


def check_shape(shape, index: int):
    """Refuses a shape Seamtag can't read either, so a file is never half used. The checks and
    their order are those of Seamtag's readShape."""
    where = f"Shape {index + 1}"
    if not isinstance(shape, dict):
        raise ValueError(f"{where} isn't an object")
    if not isinstance(shape.get("label"), str):
        raise ValueError(f"{where} needs a label")
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{where} needs points")
    if not all(isinstance(p, list) and len(p) == 2 and all(map(is_number, p)) for p in points):
        raise ValueError(f"{where} points must be [x, y] pairs")
    kind = shape_type(shape)
    if kind not in SHAPE_TYPES:
        raise ValueError(f"{where} has an unknown shape_type: {js_string(kind)}")
    if shape.get("description") is not None and not isinstance(shape["description"], str):
        raise ValueError(f"{where} description must be text")
    if shape.get("group_id") is not None and not is_whole(shape["group_id"]):
        raise ValueError(f"{where} group_id must be a whole number")
    if shape.get("mask") is not None and not isinstance(shape["mask"], str):
        raise ValueError(f"{where} mask must be a base64 PNG")
    if kind == "oriented_rectangle" and len(points) != 4:
        raise ValueError(f"{where} is an oriented rectangle, so it needs 4 points")
    # Unlike the other keys, null isn't a lock's "not set"
    if "locked" in shape and not isinstance(shape["locked"], bool):
        raise ValueError(f"{where} locked must be true or false")
    if shape.get("flags") is not None and not is_flags(shape["flags"]):
        raise ValueError(f"{where} flags must map names to true or false")


def js_string(value) -> str:
    """A JSON value as JavaScript's String() writes it, so a message that names one reads as
    Seamtag's: true, not True, and 7 for 7.0"""
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, list):
        return ",".join("" if v is None else js_string(v) for v in value)
    if isinstance(value, dict):
        return "[object Object]"
    try:
        number = float(value)
    except OverflowError:
        number = math.inf if value > 0 else -math.inf
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0:
        return "0"
    # The shortest digits that read back as the number, as both languages find them, and the
    # place of the point: the number is 0.digits times ten to that
    mantissa, _, power = repr(abs(number)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    point = len(whole) + int(power or 0) - (len(whole + fraction) - len(digits))
    digits = digits.rstrip("0")
    sign = "-" if number < 0 else ""
    if len(digits) <= point <= 21:
        return sign + digits + "0" * (point - len(digits))
    if 0 < point <= 21:
        return sign + digits[:point] + "." + digits[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * -point + digits
    rest = "." + digits[1:] if len(digits) > 1 else ""
    return f"{sign}{digits[0]}{rest}e{'+' if point > 0 else '-'}{abs(point - 1)}"


def shape_type(shape: dict) -> str:
    # A shape without one is a polygon, as Seamtag and labelme's shapes_to_label read it
    kind = shape.get("shape_type")
    return "polygon" if kind is None else kind


def is_ai(shape: dict) -> bool:
    # AI proposals only count once a person has accepted them
    return (shape.get("flags") or {}).get("ai") is True


def has_area(shape: dict) -> bool:
    """Whether a shape covers pixels: lines and points don't, nor shapes short of points"""
    kind, count = shape_type(shape), len(shape.get("points") or [])
    if kind in ("polygon", "oriented_rectangle"):
        return count >= 3
    if kind in ("rectangle", "circle"):
        return count >= 2
    return kind == "mask" and count >= 2 and bool(shape.get("mask"))


# Pascal VOC, YOLO and CreateML, read as LabelImg 1.8.6 reads them (libs/pascal_voc_io.py,
# yolo_io.py and create_ml_io.py) and as Seamtag's src/shared/formats.ts does, so a folder
# labelled in either trains here. Ported from LabelImg (https://github.com/HumanSignal/labelImg)
# under its MIT licence:
#
# Copyright (c) <2015-Present> Tzutalin
#
# Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell,
# Antonio Torralba, William T. Freeman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# JavaScript's white space, which Seamtag trims and splits on: a BOM is part of it
JS_SPACE = (
    "\t\n\v\f\r \xa0\u1680"
    + "".join(map(chr, range(0x2000, 0x200B)))
    + "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
JS_SPACES = re.compile(f"[{re.escape(JS_SPACE)}]+")
XML_NAME = re.compile(f"[^{re.escape(JS_SPACE)}/>]+")
# A tag ends at the first ">" outside quotes
XML_TAG_END = re.compile(r"""(?:[^>"']|"[^"]*"|'[^']*')*>""")
XML_ENTITY = re.compile(r"&(#x[0-9a-f]+|#[0-9]+|[a-z]+);", re.IGNORECASE | re.ASCII)
XML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}
VOC_DIFFICULT = re.compile("1|true|yes", re.IGNORECASE | re.ASCII)


def js_number(text: str) -> float:
    """A number as JavaScript's Number() reads one, or NaN, so these files read as in Seamtag"""
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text, re.ASCII):
        return float(text)
    if re.fullmatch(r"0([xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)", text):
        try:
            return float(int(text, 0))
        except OverflowError:
            return math.inf
    return math.nan


def box_shape(label: str, x0, y0, x1, y1, difficult: bool = False) -> dict:
    """A box these formats keep, as a LabelMe rectangle"""
    return {
        "label": label,
        "points": [[x0, y0], [x1, y1]],
        "shape_type": "rectangle",
        "flags": {"difficult": True} if difficult else {},
    }


def xml_unescape(text: str) -> str:
    def entity(match):
        name = match.group(1)
        if name[0] != "#":
            return XML_ENTITIES.get(name, match.group())
        code = int(name[2:], 16) if name[1] in "xX" else int(name[1:])
        return chr(code) if code <= 0x10FFFF else match.group()

    return XML_ENTITY.sub(entity, text)


def xml_skip(text: str, end: str, start: int, what: str) -> int:
    at = text.find(end, start)
    if at < 0:
        raise ValueError(f"It isn't valid XML: {what} never ends")
    return at + len(end)


def parse_xml(text: str) -> dict:
    """XML, as much as label files use: elements, text, entities, CDATA and comments. Each element
    is {"name", "children", "text"}, its text without its children's."""
    stack, root = [], None
    i = 1 if text.startswith("\ufeff") else 0
    while i < len(text):
        lt = text.find("<", i)
        chunk = text[i:] if lt < 0 else text[i:lt]
        if stack:
            stack[-1]["text"] += xml_unescape(chunk)
        elif chunk.strip(JS_SPACE):
            raise ValueError("It isn't valid XML: there's text outside the top element")
        if lt < 0:
            break
        if text.startswith("<!--", lt):
            i = xml_skip(text, "-->", lt + 4, "a comment")
        elif text.startswith("<![CDATA[", lt):
            i = xml_skip(text, "]]>", lt + 9, "a CDATA section")
            if stack:
                stack[-1]["text"] += text[lt + 9 : i - 3]
        elif text.startswith("<?", lt):
            i = xml_skip(text, "?>", lt + 2, "a declaration")
        elif text.startswith("<!", lt):
            # A DOCTYPE, with any internal subset in brackets
            i, depth = lt + 2, 0
            while i < len(text) and not (text[i] == ">" and depth <= 0):
                depth += {"[": 1, "]": -1}.get(text[i], 0)
                i += 1
            i += 1
        else:
            end = XML_TAG_END.match(text, lt + 1)
            if not end:
                raise ValueError("It isn't valid XML: a tag never ends")
            body, i = text[lt + 1 : end.end() - 1], end.end()
            if body.startswith("/"):
                name = body[1:].strip(JS_SPACE)
                if not stack or stack.pop()["name"] != name:
                    raise ValueError(f"It isn't valid XML: </{name}> closes nothing that's open")
                continue
            closed = body.endswith("/")
            name = XML_NAME.match(body[:-1] if closed else body)
            if not name:
                raise ValueError("It isn't valid XML: a tag has no name")
            element = {"name": name.group(), "children": [], "text": ""}
            if stack:
                stack[-1]["children"].append(element)
            elif root is not None:
                raise ValueError("It isn't valid XML: it has two top elements")
            else:
                root = element
            if not closed:
                stack.append(element)
    if stack:
        raise ValueError(f"It isn't valid XML: <{stack[-1]['name']}> is never closed")
    if root is None:
        raise ValueError("It isn't valid XML: it has no elements")
    return root


def xml_child(element: dict, name: str):
    return next((c for c in element["children"] if c["name"] == name), None)


def read_voc(text: str) -> list:
    """LabelImg's PascalVocReader, which keeps whole pixels; boxes with fractions keep them here"""
    root = parse_xml(text)
    if root["name"] != "annotation":
        raise ValueError(
            f"It isn't a Pascal VOC file: the top element is <{root['name']}>, not <annotation>"
        )
    shapes = []
    objects = [c for c in root["children"] if c["name"] == "object"]
    for number, obj in enumerate(objects, start=1):
        name, bndbox = xml_child(obj, "name"), xml_child(obj, "bndbox")
        if name is None:
            raise ValueError(f"Object {number} has no <name>")
        if bndbox is None:
            raise ValueError(f"Object {number} has no <bndbox>")
        corners = []
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            value = xml_child(bndbox, tag)
            value = "" if value is None else value["text"].strip(JS_SPACE)
            if not value or not math.isfinite(js_number(value)):
                raise ValueError(f"Object {number} needs a number in <{tag}>")
            corners.append(js_number(value))
        difficult = xml_child(obj, "difficult")
        text = "" if difficult is None else difficult["text"].strip(JS_SPACE)
        hard = VOC_DIFFICULT.fullmatch(text) is not None
        shapes.append(box_shape(name["text"].strip(JS_SPACE), *corners, hard))
    return shapes


def read_classes(text: str) -> list:
    """classes.txt as LabelImg reads it: blank lines inside stay, keeping the numbers after them.
    A byte order mark, which some editors write first, is left out of the first name, where
    LabelImg and Seamtag 3.2.0 keep it, so a dog is still a dog; the numbers stay the same."""
    names = re.split(r"\r?\n", re.sub(r"\A[\r\n]+|[\r\n]+\Z", "", text))
    names[0] = names[0].removeprefix("\ufeff")
    return names


def read_yolo(text: str, classes: list, width: int, height: int) -> list:
    """LabelImg's YoloReader: each line is a class and a box's centre and size, as shares of the
    image, rounded to whole pixels as Python rounds, halves to the even number"""
    shapes = []
    for number, line in enumerate(re.split(r"\r?\n", text), start=1):
        line = line.strip(JS_SPACE)
        if not line:
            continue
        parts = JS_SPACES.split(line)
        if len(parts) != 5:
            raise ValueError(f"Line {number} needs a class and four numbers, as YOLO's boxes have")
        c, xc, yc, w, h = map(js_number, parts)
        if not all(map(math.isfinite, (c, xc, yc, w, h))):
            raise ValueError(f"Line {number} has something that isn't a number")
        if not c.is_integer() or c < 0 or c >= len(classes):
            raise ValueError(f"Line {number}: class {parts[0]} isn't in {YOLO_CLASSES}")
        corners = (
            round(width * max(xc - w / 2, 0)),
            round(height * max(yc - h / 2, 0)),
            round(width * min(xc + w / 2, 1)),
            round(height * min(yc + h / 2, 1)),
        )
        shapes.append(box_shape(classes[int(c)], *corners))
    return shapes


def read_createml(text: str, image: str):
    """LabelImg's CreateMLReader: the boxes of this image's entries, which keep any fractions, or
    None when the list doesn't name the image"""
    entries = parse_json(text)
    if not isinstance(entries, list):
        raise ValueError("It isn't a CreateML file: the top level must be a list")
    shapes, found = [], False
    for number, entry in enumerate(entries, start=1):
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("image"), str)
            and isinstance(entry.get("annotations"), list)
        ):
            raise ValueError(f"Entry {number} needs an image name and a list of annotations")
        if entry["image"] != image:
            continue
        found = True
        for count, note in enumerate(entry["annotations"], start=1):
            box = note.get("coordinates") if isinstance(note, dict) else None
            if not (
                isinstance(note, dict)
                and isinstance(note.get("label"), str)
                and isinstance(box, dict)
                and all(
                    isinstance(box.get(k), (int, float)) and not isinstance(box.get(k), bool)
                    for k in ("x", "y", "width", "height")
                )
            ):
                raise ValueError(
                    f"Annotation {count} of {image} needs a label and coordinates x, y, "
                    "width and height"
                )
            # The box's centre, and its size
            x, y, w, h = (float(box[k]) for k in ("x", "y", "width", "height"))
            shapes.append(box_shape(note["label"], x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return shapes if found else None


def looks_like_createml(text: str) -> bool:
    # CreateML's files take LabelMe's name: a list is CreateML's, anything else LabelMe's
    return text.lstrip(JS_SPACE).startswith("[")


def place_mask(data: str, points: list, size: tuple):
    """Where a mask shape's PNG lands, as labelme 7.9.1 places it: at its first point, rounded.
    When both points are whole numbers they are the box's corners, both inside it, and the PNG is
    cut to that box; otherwise the PNG keeps its own size. Gives the box on the image and the
    pixels inside it (nonzero is inside), or None when none of it is on the image."""
    try:
        pixels = np.array(Image.open(io.BytesIO(base64.b64decode(data))))
    except (OSError, ValueError):
        raise ValueError("A mask shape's pixels aren't a PNG") from None
    if pixels.ndim != 2:
        raise ValueError("A mask shape's PNG must have one channel")
    (x0, y0), (x1, y1) = points[:2]
    left, top = round(x0), round(y0)
    if all(float(v).is_integer() for v in (x0, y0, x1, y1)):
        rows, cols = round(y1) - top + 1, round(x1) - left + 1
    else:
        rows, cols = pixels.shape
    width, height = size
    box = (max(left, 0), max(top, 0), min(left + cols, width), min(top + rows, height))
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    inside = np.zeros((box[3] - box[1], box[2] - box[0]), dtype=bool)
    # A PNG smaller than its box leaves the rest of the box outside
    part = pixels[box[1] - top : box[3] - top, box[0] - left : box[2] - left] != 0
    inside[: part.shape[0], : part.shape[1]] = part
    return box, Image.fromarray(inside)


def rasterize(shapes: list, size: tuple, class_ids: dict) -> np.ndarray:
    """Draw shapes into an index mask, in file order, so later shapes win. Each covers the pixels
    labelme 7.9.1's shapes_to_label gives it; lines and points have no area and draw nothing."""
    mask = Image.new("L", size, 0)
    for shape in shapes:
        class_id = class_ids.get(shape.get("label"))
        if class_id is None or not has_area(shape):
            continue
        kind = shape_type(shape)
        points = [tuple(p) for p in shape["points"]]
        draw = ImageDraw.Draw(mask)
        if kind == "mask":
            placed = place_mask(shape["mask"], points, size)
            if placed:
                mask.paste(class_id, *placed)
        elif kind == "circle":
            # The centre, then a point on the edge
            (cx, cy), (px, py) = points[:2]
            r = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=class_id)
        elif kind == "rectangle" and len(points) == 2:
            # Rectangles can be drawn in any direction, and PIL needs the top left first
            (x0, y0), (x1, y1) = points
            draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=class_id)
        else:
            # Polygons, oriented rectangles' four corners, and rectangles given by all their corners
            draw.polygon(points, fill=class_id)
    return np.array(mask)


def is_val(name: str, val_split: float) -> bool:
    """Stable split: an image stays on the same side as the dataset grows."""
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_split


def tile_starts(size: int, tile: int) -> list:
    """Where tiles start along one side: they cover all of it, the last one moved back to stay full size."""
    return [0] if size <= tile else [*range(0, size - tile, tile), size - tile]


def tile_stem(name: str) -> str:
    """The start of an image's tile names, unique across folders: line2/a.jpg gives line2%2Fa,
    and a.jpg at the top still gives a. A file name can't pass 255 bytes, so a long one keeps its
    start and ends in %- and a hash of the whole name, which no shorter one does."""
    stem = posixpath.splitext(name)[0].replace("%", "%25").replace("/", "%2F")
    if len(stem.encode()) <= TILE_STEM_BYTES:
        return stem
    start = stem.encode()[: TILE_STEM_BYTES - 20].decode(errors="ignore")
    return f"{start}%-{hashlib.sha1(name.encode()).hexdigest()[:12]}"


# By profile and mode: how images in it come to sRGB, or None when they already are
SRGB_TRANSFORMS = {}


def srgb_transform(profile: bytes, mode: str):
    source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    transform = ImageCms.buildTransform(source, ImageCms.createProfile("sRGB"), mode, "RGB")
    if mode != "RGB":
        return transform
    # Most photos come in sRGB, which changes nothing but takes a second or two for a big one
    steps = np.arange(0, 256, 15, dtype=np.uint8)
    probe = np.stack(np.meshgrid(steps, steps, steps), -1).reshape(1, -1, 3)
    out = np.asarray(ImageCms.applyTransform(Image.fromarray(probe), transform))
    return None if np.abs(out.astype(int) - probe).max() <= 1 else transform


def to_srgb(image: Image.Image, profile: bytes) -> Image.Image:
    """Colours through the image's own ICC profile, as libvips gives them to Seamtag and to the
    window: photos in Adobe RGB, Display P3 or CMYK differ from sRGB. A profile that can't be used
    is left out, as libvips leaves it out."""
    if ImageCms is None:
        return image
    if image.mode not in ("RGB", "CMYK", "L"):
        # Alpha goes, as Seamtag drops it
        image = image.convert("L" if image.mode in ("1", "LA") else "RGB")
    try:
        key = (profile, image.mode)
        if key not in SRGB_TRANSFORMS:
            SRGB_TRANSFORMS[key] = srgb_transform(profile, image.mode)
        transform = SRGB_TRANSFORMS[key]
        return ImageCms.applyTransform(image, transform) if transform else image
    except (OSError, ValueError, ImageCms.PyCMSError):
        return image


def open_image(path: Path) -> Image.Image:
    """The image upright and in 8-bit sRGB, as Seamtag's pre-labelling reads it: 16-bit greys keep
    their high byte, where converting them straight to RGB would cut them off at white, and other
    images go through their ICC profile. CMYK without a profile is converted as Pillow converts it,
    where libvips uses a CMYK profile of its own, so its colours differ between the apps."""
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode.startswith("I;16"):
        # libvips doesn't apply a profile to 16-bit greys either
        return Image.fromarray((np.asarray(image) >> 8).astype(np.uint8)).convert("RGB")
    profile = image.info.get("icc_profile")
    if profile:
        image = to_srgb(image, profile)
    return image.convert("RGB")


def read_labels(
    root: Path, image: Path, file: Path, size: tuple = (1, 1), texts: dict = None
) -> tuple:
    """read_label_file, stopping everything with the file's name when it can't be read"""
    try:
        return read_label_file(file, image, size, texts)
    except (OSError, ValueError, ArithmeticError) as error:
        raise SystemExit(f"{file.relative_to(root).as_posix()} can't be read: {error}")


def prepare(root: Path, val_split: float, tile: int) -> dict:
    project = load_project(root)
    class_ids = {c["name"]: i + 1 for i, c in enumerate(project["classes"])}
    out = root / ".myseg" / "dataset"
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "masks").mkdir(parents=True)

    labelled, unlabelled, tiles = [], 0, {"train": 0, "val": 0}
    pixels = np.zeros(len(class_ids) + 1, dtype=np.int64)
    unknown, skipped, formats = set(), Counter(), Counter()
    for path in list_images(root):
        file = find_labels(path)
        # A YOLO file and its classes.txt are kept, to read them again at the image's size as
        # they were here, even if Seamtag changes them before then
        texts = {} if file and file.suffix == ".txt" else None
        fmt, shapes = read_labels(root, path, file, texts=texts) if file else (None, [])
        # AI proposals only count once a person has accepted them
        shapes = [s for s in shapes if not is_ai(s)]
        # Lines and points have no area to train on: counted, so the app can say so
        skipped.update(shape_type(s) for s in shapes if not has_area(s))
        shapes = [s for s in shapes if has_area(s)]
        # Labels not in the class list are named, to add them, and until then their shapes are
        # left out, as if not drawn: an image with only those isn't labelled, not all background
        unknown |= {s["label"] for s in shapes} - set(class_ids)
        if not any(s["label"] in class_ids for s in shapes):
            unlabelled += 1
            continue
        formats[fmt] += 1
        labelled.append((path, path.relative_to(root).as_posix(), file, shapes, texts))

    # Split by the name without its extension: at the top of the folder that's the file's stem,
    # as it always was, so projects keep their split
    def stem(name):
        return posixpath.splitext(name)[0]

    # Small sets can land all on one side; both sides need at least one image
    forced = {}
    if len(labelled) >= 2:
        sides = [is_val(stem(item[1]), val_split) for item in labelled]
        if all(sides):
            forced[labelled[0][1]] = "train"
        elif not any(sides):
            forced[labelled[-1][1]] = "val"

    for path, name, file, shapes, texts in labelled:
        split = forced.get(name) or ("val" if is_val(stem(name), val_split) else "train")
        try:
            image = open_image(path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"{name} can't be opened: {error}")
        if texts is not None:
            # YOLO's boxes are shares of the image: read again at its upright size
            shapes = [s for s in read_labels(root, path, file, image.size, texts)[1] if has_area(s)]
        try:
            mask = rasterize(shapes, image.size, class_ids)
        except (OSError, ValueError, ArithmeticError) as error:
            raise SystemExit(f"{file.relative_to(root).as_posix()} can't be read: {error}")
        pixels += np.bincount(mask.ravel(), minlength=len(pixels))[: len(pixels)]
        width, height = image.size
        for top in tile_starts(height, tile):
            for left in tile_starts(width, tile):
                box = (left, top, min(left + tile, width), min(top + tile, height))
                piece = mask[box[1] : box[3], box[0] : box[2]]
                # Skip empty tiles of big images, or background swamps the set
                if max(width, height) > tile and not piece.any():
                    continue
                tile_name = f"{tile_stem(name)}__{left}_{top}.png"
                try:
                    image.crop(box).save(out / split / "images" / tile_name)
                    Image.fromarray(piece).save(out / split / "masks" / tile_name)
                except OSError as error:
                    raise SystemExit(f"{name}'s tiles can't be saved: {error}")
                tiles[split] += 1

    summary = {
        "labelled": len(labelled),
        "unlabelled": unlabelled,
        "train": tiles["train"],
        "val": tiles["val"],
        "unknown_labels": sorted(l for l in unknown if l),
        "class_pixels": {
            name: int(pixels[i]) for name, i in {"background": 0, **class_ids}.items()
        },
        "skipped_shapes": dict(sorted(skipped.items())),
        "formats": {f: formats[f] for f in LABEL_FORMATS},
    }
    (out / "summary.json").write_text(json.dumps(summary))
    return summary


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def train(root: Path, options: dict, run: str = None, prepared: bool = False) -> int:
    """Train one run. With prepared=True the dataset was built and uploaded by the app."""
    root = root.resolve()
    run_dir = root / ".myseg" / "runs" / (run or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "options.json").write_text(json.dumps(options, indent=2))
    events = run_dir / "events.jsonl"

    def emit(**event):
        with events.open("a") as f:
            f.write(json.dumps(event) + "\n")

    print(json.dumps({"run": run_dir.name}), flush=True)
    if prepared:
        summary = json.loads((root / ".myseg" / "dataset" / "summary.json").read_text())
    else:
        try:
            summary = prepare(root, options.get("val_split", 0.2), options.get("tile", 1024))
        except SystemExit as error:
            # Kept with the run, so it still says why it stopped once the live log is gone
            emit(type="error", message=str(error))
            raise
    emit(type="prepared", **summary)
    if summary["train"] == 0 or summary["val"] == 0:
        emit(type="error", message="Need labelled images for both train and validation.")
        return 1

    project = load_project(root)
    config = copy.deepcopy(yaml.safe_load((REPO / "configs" / "default.yaml").read_text()))
    num_classes = len(project["classes"]) + 1
    config["experiment"]["name"] = "run"
    config["data"].update(
        base_dir=str(root / ".myseg"),
        dataset_dir="dataset",
        image_size=options["image_size"],
        num_classes=num_classes,
    )
    config["model"].update(
        architecture=options["architecture"],
        encoder=options["encoder"],
        num_classes=num_classes,
        encoder_weights=options.get("encoder_weights", "imagenet"),
    )
    config["training"].update(
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        learning_rate=options["learning_rate"],
        num_workers=options.get("num_workers", 2),
        gradient_accumulation_steps=1,
    )
    if config["training"]["num_workers"] == 0:
        config["training"].update(persistent_workers=False, prefetch_factor=None)
    config["loss"]["type"] = options["loss"]
    config["loss"]["class_weights"] = [1.0] * num_classes
    config["logging"].update(wandb=False, tensorboard=False)
    config["paths"].update(
        output_dir=str(run_dir), checkpoint_dir=str(run_dir), log_dir=str(run_dir)
    )
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # The run keeps its classes, so its model's outputs keep their names when the project's change
    (run_dir / "classes.json").write_text(json.dumps([c["name"] for c in project["classes"]]))

    emit(type="start", epochs=options["epochs"], device=device_name())
    command = [
        sys.executable, str(REPO / "scripts" / "train.py"),
        "--config", str(config_path), "--events", str(events),
    ]
    code = subprocess.call(command, cwd=REPO)
    if code != 0:
        emit(type="error", message=f"Training stopped with exit code {code}.")
    return code


def load_model(run_dir: Path, run: str, device: str) -> tuple:
    """The run's best model, built as training built it but giving raw scores, and its config"""
    import torch

    from src.models import create_model

    checkpoints = sorted(run_dir.glob("run_*/best_model.pth"))
    if not checkpoints:
        raise SystemExit(f"Run {run} has no saved model yet.")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model_cfg = config["model"]
    model = create_model(
        architecture=model_cfg["architecture"],
        encoder=model_cfg["encoder"],
        encoder_weights=None,
        num_classes=model_cfg["num_classes"],
        activation=None,
    )
    # The newest, should a run hold more than one
    state = torch.load(checkpoints[-1], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def run_classes(root: Path, run_dir: Path) -> list:
    """The names of the classes a run learned, background left out: its own list, or for runs
    from before runs kept one, the project's"""
    saved = run_dir / "classes.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    return [c["name"] for c in load_project(root)["classes"]]


def predict_labels(model, transform, image: np.ndarray, tile: int, device: str) -> tuple:
    """Class index and confidence (0-255) per pixel. Big images go tile by tile, as in training."""
    import cv2
    import torch

    height, width = image.shape[:2]
    labels = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.uint8)
    step = tile if max(height, width) > tile else max(height, width)
    with torch.inference_mode():
        for top in tile_starts(height, step):
            for left in tile_starts(width, step):
                piece = image[top : top + step, left : left + step]
                batch = transform(image=piece)["image"].unsqueeze(0).to(device)
                best, index = model(batch).softmax(1).max(1)
                size = (piece.shape[1], piece.shape[0])
                tile_labels = cv2.resize(
                    index[0].cpu().numpy().astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
                )
                tile_confidence = cv2.resize(
                    (best[0].cpu().numpy() * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR
                )
                # Tiles overlap near the edges: keep whichever is surer
                region = (slice(top, top + step), slice(left, left + step))
                surer = tile_confidence > confidence[region]
                labels[region][surer] = tile_labels[surer]
                confidence[region][surer] = tile_confidence[surer]
    return labels, confidence


def shapes_from_labels(labels: np.ndarray, confidence: np.ndarray, classes: list, min_area: int) -> list:
    """Polygons for each class. Score is the mean confidence inside the shape: use it to rank, not as a probability."""
    import cv2

    shapes = []
    for index, cls in enumerate(classes, start=1):
        contours, _ = cv2.findContours(
            (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            points = cv2.approxPolyDP(contour, epsilon, True)[:, 0].tolist()
            if len(points) < 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            inside = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inside, [contour], -1, 1, -1, offset=(-x, -y))
            score = confidence[y : y + h, x : x + w][inside > 0].mean() / 255
            shapes.append(
                {
                    "label": cls["name"],
                    "points": points,
                    "shape_type": "polygon",
                    "group_id": None,
                    "description": "",
                    "flags": {"ai": True},
                    "score": round(float(score), 3),
                }
            )
    return shapes


def predict(root: Path, run: str, images: list, min_area: int):
    """Yield the device once, then one result per image. A bad image doesn't stop the rest."""
    import torch

    from src.data.transforms import get_val_transforms

    run_dir = root / ".myseg" / "runs" / run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(run_dir, run, device)

    size = config["data"]["image_size"]
    normalize = config["augmentation"]["normalize"]
    transform = get_val_transforms(size, size, **normalize)
    tile = json.loads((run_dir / "options.json").read_text()).get("tile", 1024)
    classes = [{"name": name} for name in run_classes(root, run_dir)]
    yield {"device": device}
    for name in images:
        try:
            image = np.array(open_image(root / name))
            labels, confidence = predict_labels(model, transform, image, tile, device)
            yield {
                "name": name,
                "shapes": shapes_from_labels(labels, confidence, classes, min_area),
                "size": [image.shape[1], image.shape[0]],
            }
        except Exception as error:
            yield {"name": name, "error": str(error)}


def per_channel(value, default: float) -> list:
    # One value per colour, as albumentations spreads a single one over all three
    values = [default] if value is None else value if isinstance(value, (list, tuple)) else [value]
    return [float(v) for v in (list(values) * 3 if len(values) == 1 else values)]


def seamtag_normalize(normalize: dict) -> tuple:
    """The mean and std Seamtag needs to feed the model what training fed it. Seamtag gives
    (pixel / 255 - mean) / std, and albumentations' Normalize, with the run's settings and
    get_val_transforms' defaults, gives (pixel - mean * max_pixel_value) / (std * max_pixel_value)
    """
    scale = normalize.get("max_pixel_value", 255.0) / 255
    mean = [v * scale for v in per_channel(normalize.get("mean"), 0.0)]
    std = [v * scale for v in per_channel(normalize.get("std"), 1.0)]
    return mean, std


def write_atomic(path: Path, text: str):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def export_classes(root: Path, run_dir: Path, run: str, num_classes: int) -> list:
    """The names the sidecar gives the model's classes. Runs from before runs kept their own take
    the project's, but only while there are as many as the model has."""
    classes = run_classes(root, run_dir)
    if len(classes) == num_classes - 1:
        return classes
    if (run_dir / "classes.json").exists():
        raise SystemExit(
            f"Run {run}'s classes.json names {len(classes)} classes, "
            f"but its model has {num_classes - 1}."
        )
    raise SystemExit(
        f"Run {run} doesn't say which classes it learned, and the project's {len(classes)} "
        f"can't be them: its model has {num_classes - 1}."
    )


def declare_output(file: Path, shape: tuple):
    """The exporter's guesses at shapes inside the graph can take a side of the image for the
    batch, as they do in Unet: the output is declared as it is, and the guesses are dropped."""
    import onnx

    model = onnx.load(str(file))
    del model.graph.value_info[:]
    dims = model.graph.output[0].type.tensor_type.shape
    dims.ClearField("dim")
    dims.dim.add().dim_param = "batch_size"
    for side in shape[1:]:
        dims.dim.add().dim_value = side
    onnx.save(model, str(file))


def check_runtime(file: Path, sample: np.ndarray, scores: np.ndarray):
    """The exported model must score as PyTorch does, where onnxruntime is there to ask. Rounding
    grows with the scores, which reach tens of thousands in PAN, so the gap allowed is 0.001 of
    the largest score, and never under 0.001."""
    try:
        import onnxruntime
    except ImportError:
        return
    session = onnxruntime.InferenceSession(str(file), providers=["CPUExecutionProvider"])
    gap = float(np.abs(session.run(None, {"input": sample})[0] - scores).max())
    if not gap <= 1e-3 * max(1.0, float(np.abs(scores).max())):
        raise SystemExit(
            f"The exported model's scores are {gap:.2g} off PyTorch's, so it wasn't saved."
        )


@contextlib.contextmanager
def exportable_pools():
    """torch's ONNX exporter writes an adaptive average pool, other than a global one, only when
    its cells are all one size and it knows the input's size. PSPNet's pools miss one or the
    other: 3 x 3 and 6 x 6 cells over the 64 x 64 features of a 512 image, and any pool behind
    an EfficientNet encoder. While exporting, each is two products with fixed matrices instead,
    which average the same pixels."""
    import torch
    from torch.nn import functional

    pool = functional.adaptive_avg_pool2d

    def cells(size, count):
        # Cell i covers floor(i * size / count) to ceil((i + 1) * size / count), as in PyTorch.
        # Models export in float32.
        weights = np.zeros((count, size), dtype=np.float32)
        for i in range(count):
            start, end = i * size // count, -(-(i + 1) * size // count)
            weights[i, start:end] = 1 / (end - start)
        return torch.from_numpy(weights)

    def fixed_pool(x, output_size):
        rows, cols = output_size if isinstance(output_size, (list, tuple)) else (output_size,) * 2
        if (rows, cols) == (1, 1):
            return pool(x, output_size)
        # The export's image size is fixed: the sides are numbers, and the matrices constants
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            height, width = (int(side) for side in x.shape[-2:])
            rows, cols = height if rows is None else rows, width if cols is None else cols
            return cells(height, rows) @ x @ cells(width, cols).T

    functional.adaptive_avg_pool2d = fixed_pool
    try:
        yield
    finally:
        functional.adaptive_avg_pool2d = pool


def export(root: Path, run: str) -> dict:
    """Write the run's model as ONNX, and beside it the sidecar Seamtag's pre-labelling reads.
    Each file appears whole, the sidecar last, so Seamtag never lists half a model."""
    import onnx
    import torch

    root = Path(os.path.abspath(root))
    run_dir = root / ".myseg" / "runs" / run
    model, config = load_model(run_dir, run, "cpu")
    classes = export_classes(root, run_dir, run, config["model"]["num_classes"])
    size = config["data"]["image_size"]
    options_file = run_dir / "options.json"
    options = json.loads(options_file.read_text()) if options_file.exists() else {}
    mean, std = seamtag_normalize(config["augmentation"]["normalize"])
    sidecar = {
        "classes": classes,
        "mean": mean,
        "std": std,
        "tile": options.get("tile", 1024),
        "input_size": [size, size],
    }

    model_file, sidecar_file = run_dir / "model.onnx", run_dir / "model.json"
    temp = run_dir / f".model.onnx.{os.getpid()}.tmp"
    # Pixels as Seamtag feeds them, (pixel / 255 - mean) / std, to export with and check on
    pixels = torch.randint(0, 256, (1, 3, size, size), generator=torch.Generator().manual_seed(0))
    sample = (pixels / 255 - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    # Newer PyTorch defaults to an exporter that needs onnxscript: keep the TorchScript one, as
    # scripts/export_onnx.py does
    accepted = inspect.signature(torch.onnx.export).parameters
    kwargs = {"dynamo": False} if "dynamo" in accepted else {}
    try:
        # Only the result goes to stdout, where the app reads it
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            scores = model(sample).numpy()
            try:
                with exportable_pools():
                    torch.onnx.export(
                        model,
                        sample,
                        str(temp),
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        # Any number of images at once, each the size the model trained on
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                        **kwargs,
                    )
            except Exception as error:
                # The exporter's messages can hold the whole graph, and the app shows one line
                reason = (str(error).strip().splitlines() or [type(error).__name__])[0]
                raise SystemExit(f"Run {run}'s model can't be written as ONNX: {reason}") from None
            declare_output(temp, scores.shape)
            onnx.checker.check_model(str(temp))
            check_runtime(temp, sample.numpy(), scores)
        os.replace(temp, model_file)
    finally:
        temp.unlink(missing_ok=True)
    write_atomic(sidecar_file, json.dumps(sidecar, indent=2))
    return {
        "model": str(model_file),
        "sidecar": str(sidecar_file),
        "classes": classes,
        "input_size": [size, size],
        "tile": sidecar["tile"],
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("project", type=Path)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--tile", type=int, default=1024)
    p = sub.add_parser("train")
    p.add_argument("project", type=Path)
    p.add_argument("--options", required=True, help="JSON training options")
    p.add_argument("--run", help="Run name; defaults to the current time")
    p.add_argument("--prepared", action="store_true", help="Use the dataset already in place")
    p = sub.add_parser("predict")
    p.add_argument("project", type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--images", nargs="+", help="Image names; without it, a JSON list on stdin")
    p.add_argument("--min-area", type=int, default=25)
    p = sub.add_parser("export")
    p.add_argument("project", type=Path)
    p.add_argument("--run", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare(args.project, args.val_split, args.tile)))
    elif args.command == "train":
        sys.exit(train(args.project, json.loads(args.options), args.run, args.prepared))
    elif args.command == "export":
        print(json.dumps(export(args.project, args.run)), flush=True)
    else:
        images = args.images or json.load(sys.stdin)
        for result in predict(args.project, args.run, images, args.min_area):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
