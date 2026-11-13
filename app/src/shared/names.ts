/**
 * File names as the system tells them apart, for finding labels in a folder's listing, as Seamtag does:
 * Windows and macOS ignore case, and macOS also the Unicode form a name is in. Finder writes Hangul
 * and accented letters decomposed (NFD), where most other apps write them composed (NFC).
 */
export function nameFold(platform: string): (name: string) => string {
  if (platform === 'darwin') return (name) => name.normalize('NFC').toLowerCase()
  if (platform === 'win32') return (name) => name.toLowerCase()
  return (name) => name
}
