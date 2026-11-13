export type LabelClass = { name: string; color: string }

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** The classes a project file lists, or null when it has no list to read */
export function classesIn(doc: unknown): LabelClass[] | null {
  if (!isObject(doc) || !Array.isArray(doc.classes)) return null
  return doc.classes.filter((c) => typeof c?.name === 'string').map((c) => ({
    name: c.name, color: typeof c.color === 'string' ? c.color : ''
  }))
}

/**
 * Seamline's class list (ours) merged by name with the file's as it is now (theirs), given the list as
 * Seamline last read or wrote it (base): what another app, such as Seamtag, added, recoloured or
 * deleted since stays so, unless Seamline changed that class too. With no file to read, ours goes as it is.
 */
export function mergeClasses(base: LabelClass[], ours: LabelClass[], theirs: LabelClass[] | null): LabelClass[] {
  if (!theirs) return ours
  const before = new Map(base.map((c) => [c.name, c.color]))
  const now = new Map(theirs.map((c) => [c.name, c.color]))
  const merged: LabelClass[] = []
  for (const c of ours) {
    const color = now.get(c.name)
    // Added or recoloured here: ours
    if (before.get(c.name) !== c.color) merged.push(c)
    // Left as it was here, and deleted or recoloured there
    else if (color !== undefined) merged.push(color === c.color ? c : { ...c, color })
  }
  // Added there: once, as Seamtag merges, even if the file names it twice
  const named = new Set(ours.map((c) => c.name))
  for (const c of theirs) {
    if (before.has(c.name) || named.has(c.name)) continue
    named.add(c.name)
    merged.push(c)
  }
  return merged
}

/** .myseg/project.json as Seamline writes it: the file as it is now, every key kept, with the classes merged */
export function projectFile(base: LabelClass[], ours: LabelClass[], onDisk: unknown, name: string) {
  const classes = mergeClasses(base, ours, classesIn(onDisk))
  return isObject(onDisk) ? { ...onDisk, classes } : { name, classes }
}
