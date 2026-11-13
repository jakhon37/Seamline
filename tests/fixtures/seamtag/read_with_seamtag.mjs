// What Seamtag finds and reads for each image in a folder, as JSON, for
// tests/unit/test_studio_formats.py: Seamline must find and read the same labels. It runs
// Seamtag's own findLabels and readLabels (src/main/project.ts), so it's bundled from a Seamtag
// checkout with Seamtag's esbuild and run with its Node; see README.md.
import { readFileSync } from 'fs'
import path from 'path'
import { findLabels, openProject, readLabels } from 'seamtag/src/main/project'

// YOLO's boxes need the image's size. These are PNGs without EXIF, so it's in their header.
const pngSize = (file) => {
  const head = readFileSync(file)
  return [head.readUInt32BE(16), head.readUInt32BE(20)]
}

const dir = path.resolve(process.argv[2])
const { images } = await openProject(dir, null)
const out = {}
for (const { name } of images) {
  let format = null
  try {
    const found = await findLabels(dir, name)
    format = found?.format ?? null
    if (!found) {
      out[name] = { format }
      continue
    }
    const labels = await readLabels(dir, name, found, async () => pngSize(path.join(dir, name)))
    // A CreateML list that doesn't name the image gives null: no labels of it
    const shapes = labels && labels.doc.shapes.map((s) => ({ label: s.label, shape_type: s.shape_type, points: s.points, flags: s.flags }))
    out[name] = { format, shapes }
  } catch (e) {
    out[name] = { format, error: e.message }
  }
}
// Anything past ASCII escaped, so a label that starts with a byte order mark shows it
const ascii = (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`
process.stdout.write(JSON.stringify(out, null, 1).replace(/[^\x00-\x7e]/g, ascii) + '\n')
