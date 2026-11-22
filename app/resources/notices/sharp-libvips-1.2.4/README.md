# The libraries in libvips, sharp-libvips 1.2.4

sharp loads libvips from the @img packages of sharp-libvips 1.2.4, and on Windows from sharp's own @img/sharp-win32-x64 0.34.5. Those packages list the libraries built into libvips and name their licences, but carry none of their licence texts. This folder holds those texts, as each library's source carries them at the version in the packages' `versions.json`. `scripts/notices.mjs` writes them into the libvips part of THIRD-PARTY-NOTICES.txt.

- Each folder is named after a library's key in `versions.json`, and holds files at their paths in that library's source.
- `sources.json` gives each library's name, version and source archive, as sharp-libvips 1.2.4's `build/posix.sh` downloads it (https://github.com/lovell/sharp-libvips/blob/v1.2.4/build/posix.sh). The archives were downloaded on 13 November 2026.
- The Windows DLLs come from libvips's build-win64-mxe at v8.17.3 (sharp-libvips's `build/win.sh`). Their `versions.json` gives the same versions.

Besides each library's own licence files, the folder holds the licences of code from other projects that the build compiles in:

- libnsgif, in libvips: `libvips/foreign/libnsgif/COPYING`.
- fastfeat, vector, x86inc and SVT-AV1, in aom: `third_party/*/LICENSE*`. x86inc and SVT-AV1 are compiled in on x64 only. sharp-libvips's build of aom leaves out its copies of libyuv, libwebm and highway.
- The BDF and PCF drivers, in FreeType: `src/bdf/README` and `src/pcf/README`, which FreeType's `LICENSE.TXT` points to.
- Microsoft's data that harfbuzz builds its USE tables from: `src/ms-use/COPYING`.

Files ending in `.excerpt` hold lines of a source file's opening comment; `sources.json` gives the lines.

- libvips: the Radiance Software License in `libvips/colour/rad2float.c` and `float2rad.c`, and Sam Leffler's notice in `libvips/foreign/tiff2vips.c` and `vips2tiff.c`.
- libxml2: the notices of `dict.c` and `list.c`, which its `Copyright` file points to.

Two more details:

- glib's `COPYING` is a link in its source, to `LICENSES/LGPL-2.1-or-later.txt`. This folder holds that text as `glib/COPYING`.
- libvips's copyright line, which THIRD-PARTY-NOTICES.txt shows, is the one libvips's source files carry most often: "Copyright (C) 1991-2005 The National Gallery", in 62 of them. Others name their own authors, or none.

Not collected here:

- Notices that sit only in the opening comments of other source files.
- The Rust crates that librsvg builds in. librsvg's build fetches them from crates.io, and the `Cargo.lock` in its source names them.

When sharp moves to another sharp-libvips version, or the libraries in `versions.json` change, the script stops. It runs again once a folder named after that version holds each library's files, and its `sources.json` matches `versions.json`. sharp-libvips's `build/posix.sh` at that version says where each source comes from.
