# Data

Datasets go here and stay out of version control; only this README and `.gitkeep` are tracked.
Every file starts with an 8-byte header, a little-endian `u32` row count and a `u32` width, followed by row-major data.
Vectors use `.fbin`, `.u8bin`, `.i8bin` or `.b1bin`, ground-truth neighbors use `.ibin`, and optional vector keys use `.i32bin`.
Large datasets can be split into shards of the same format.
`retri-generate` and the `retri-download-*` binaries write files in this format.
