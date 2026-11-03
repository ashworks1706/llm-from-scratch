// a stored format changes over time, so every file starts with fields that never change:
// a version and a name. read that header first, from a prefix of the bytes, then dispatch.
// the current version decodes strictly (trailing bytes are corruption). an old version gets
// its own error naming the collection so the operator knows what to migrate. an unknown
// newer version is also its own error. never guess at a layout you do not know.
//
// from piramid: apps/engine/database/src/storage/manifest.rs (CollectionMetadata::decode,
// ManifestHeader) and apps/engine/database/src/storage/codec.rs (decode_prefix vs decode).
// toy layout, little endian: u32 version, u32 name_len, name bytes, then for v2 only:
// u8 metric (0 cosine, 1 euclidean, 2 dot) and u64 vector_count.
// explain: why a separate prefix decode instead of trying the full struct and catching the error?
// explain: why a distinct Legacy error instead of upgrading v1 in place?

#[derive(Debug, PartialEq)]
enum DecodeError { Corrupt(String), Legacy { name: String }, Unsupported { name: String, found: u32 } }

#[derive(Debug, PartialEq)]
struct Manifest { name: String, metric: u8, vector_count: u64 }

const SCHEMA_VERSION: u32 = 2;
const LEGACY_VERSION: u32 = 1;

fn decode_header(bytes: &[u8]) -> Result<(u32, String, usize), DecodeError> {
    // return (version, name, bytes consumed). truncated input or invalid utf8 is Corrupt.
    // must work for any version and must ignore whatever follows the header.
    todo!()
}

fn decode_manifest(bytes: &[u8]) -> Result<Manifest, DecodeError> {
    // header first, then: v2 -> read metric and vector_count, error if bytes are short or left over;
    // v1 -> Legacy { name }; anything else -> Unsupported { name, found }. unknown metric byte is Corrupt.
    todo!()
}
