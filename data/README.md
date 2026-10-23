# Data

Datasets go here and stay out of version control; only this README and `.gitkeep` are tracked.
Every file starts with an 8-byte header, a little-endian `u32` row count and a `u32` width, followed by row-major data.
Vectors use `.fbin`, `.u8bin`, `.i8bin` or `.b1bin`, ground-truth neighbors use `.ibin`, and optional vector keys use `.i32bin`.
`retri-generate` and the `retri-download-*` binaries write files in this format.

## Downloads

`scripts/download.sh <name>` fetches a public dataset into `data/<name>/`, resuming partial downloads, under the file names the commands below use.
Sources that ship vectors without ground truth have their own binaries behind `--features download`.
`retri-download-molecules` and `retri-download-cohere` fetch the Parquet shards from S3 and Hugging Face, extract the bit-packed column straight into `.b1bin`, sample queries with a fixed seed, and compute exact brute-force Hamming top-K with NumKong's SIMD kernels.

Large datasets can be split into shards of the same format.
`--base-vectors`, `--query-vectors`, `--query-neighbors` and `--base-keys` accept shell glob patterns, so a sharded dataset reads like a single file: pass `--base-vectors 'base.shard_*.fbin'`, quoted so the shell does not expand it.
Matched shards are natural-sorted, `shard_2.fbin` before `shard_10.fbin`, and must agree on width and scalar format.

## Unum UForm Wiki

Image-and-text embeddings from the [UForm][unum-wiki-1m] small multimodal model, projected to a shared 256d space.
Bench against IP, since UForm is L2-normalised at training time.

<details>
<summary>1M — f32, 256d, IP, ~1 GB</summary>

```sh
scripts/download.sh wiki-1m
retri-eval-usearch \
    --base-vectors data/wiki-1m/base.1M.fbin \
    --query-vectors data/wiki-1m/query.public.100K.fbin \
    --query-neighbors data/wiki-1m/groundtruth.public.100K.ibin \
    --data-type f32,f16,i8 --metric ip \
    --output results/wiki-1m
```

</details>

## Arxiv with E5

[Arxiv][unum-arxiv-2m] abstracts embedded with the `intfloat/e5-base-v2` model.
The 2.3M queries are the title embeddings of the same papers, and the ground truth records the top-100 abstracts by IP for each title.
A title's nearest abstract is rarely its own paper's, so this is not a self-search.

<details>
<summary>2.3M — f32, 768d, IP, ~7 GB</summary>

```sh
scripts/download.sh arxiv-2m
retri-eval-usearch \
    --base-vectors data/arxiv-2m/abstract.e5-base-v2.fbin \
    --query-vectors data/arxiv-2m/query.fbin \
    --query-neighbors data/arxiv-2m/groundtruth.ibin \
    --data-type f32,bf16,f16,i8 --metric ip \
    --output results/arxiv-2m
```

</details>

## Meta BIGANN SIFT

Billion-scale [SIFT][meta-bigann] descriptors from Meta.
No pre-sliced subset base files exist, so `scripts/download.sh` range-requests a prefix of the single 1B file and patches the row count in its header.
Pre-computed ground truth is available for the 10M and 100M prefixes and the full 1B set, each exact L2 top-100 over that prefix for the same 10K queries.
Those files append the `f32` distances after the neighbor IDs, so they are twice the size of an IDs-only `.ibin`.

<details>
<summary>10M — u8, 128d, L2, ~1.2 GB</summary>

```sh
scripts/download.sh sift-10m
retri-eval-usearch \
    --base-vectors data/sift-10m/base.10M.u8bin \
    --query-vectors data/sift-10m/query.public.10K.u8bin \
    --query-neighbors data/sift-10m/groundtruth.public.10K.ibin \
    --data-type f32,f16,i8 --metric l2 \
    --output results/sift-10m
```

</details>

<details>
<summary>100M — u8, 128d, L2, ~12 GB</summary>

```sh
scripts/download.sh sift-100m
retri-eval-usearch \
    --base-vectors data/sift-100m/base.100M.u8bin \
    --query-vectors data/sift-100m/query.public.10K.u8bin \
    --query-neighbors data/sift-100m/groundtruth.public.10K.ibin \
    --data-type f32,f16,i8 --metric l2 \
    --steps 20 --output results/sift-100m
```

</details>

<details>
<summary>1B — u8, 128d, L2, ~119 GB</summary>

```sh
scripts/download.sh sift-1b
```

</details>

## Microsoft Turing-ANNS

[Turing-ANNS][msft-turing] holds 373 GB of f32 vectors with 100 dimensions at full 1B scale.
Subsets are prefixes of the 1B file, sliced the same way as SIFT.
Pre-computed ground truth is available for 1M, 10M, and 100M.

<details>
<summary>1M — f32, 100d, L2, ~400 MB</summary>

```sh
scripts/download.sh turing-1m
retri-eval-usearch \
    --base-vectors data/turing-1m/base.1M.fbin \
    --query-vectors data/turing-1m/query.public.100K.fbin \
    --query-neighbors data/turing-1m/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --output results/turing-1m
```

</details>

<details>
<summary>10M — f32, 100d, L2, ~3.7 GB</summary>

```sh
scripts/download.sh turing-10m
retri-eval-usearch \
    --base-vectors data/turing-10m/base.10M.fbin \
    --query-vectors data/turing-10m/query.public.100K.fbin \
    --query-neighbors data/turing-10m/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --output results/turing-10m
```

</details>

<details>
<summary>100M — f32, 100d, L2, ~37 GB</summary>

```sh
scripts/download.sh turing-100m
retri-eval-usearch \
    --base-vectors data/turing-100m/base.100M.fbin \
    --query-vectors data/turing-100m/query.public.100K.fbin \
    --query-neighbors data/turing-100m/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric l2 \
    --steps 20 --output results/turing-100m
```

</details>

## Microsoft SpaceV

[SpaceV][msft-spacev] holds web-search embeddings already quantised to int8 at the source.
A 100M subset is mirrored on Hugging Face; the original 1B lives on AWS S3.

<details>
<summary>100M — i8, 100d, L2, ~9.3 GB</summary>

```sh
scripts/download.sh spacev-100m
retri-eval-usearch \
    --base-vectors data/spacev-100m/base.100M.i8bin \
    --query-vectors data/spacev-100m/query.30K.i8bin \
    --query-neighbors data/spacev-100m/groundtruth.30K.i32bin \
    --data-type f32,f16,i8 --metric l2 \
    --steps 20 --output results/spacev-100m
```

</details>

## Yandex Deep

[Deep][yandex-deep] holds image embeddings extracted from the GoogLeNet penultimate layer.
Only the full 1B is included here, as the smaller subsets repeat the same distribution at scales other datasets already cover.
Its ground truth is published against the full 1B only, so a slice can be judged only by `--self-search`.

<details>
<summary>1B — f32, 96d, L2, ~358 GB</summary>

```sh
scripts/download.sh deep-1b
```

</details>

## Yandex Text-to-Image

[Text-to-Image][yandex-t2i] holds cross-modal text-and-image embeddings, benchmarked under cosine similarity.

<details>
<summary>1M — f32, 200d, Cos, ~750 MB</summary>

```sh
scripts/download.sh t2i-1m
retri-eval-usearch \
    --base-vectors data/t2i-1m/base.1M.fbin \
    --query-vectors data/t2i-1m/query.public.100K.fbin \
    --query-neighbors data/t2i-1m/groundtruth.public.100K.ibin \
    --data-type f32,bf16,f16,i8 --metric cos \
    --output results/t2i-1m
```

</details>

<details>
<summary>1B — f32, 200d, Cos, ~750 GB</summary>

```sh
scripts/download.sh t2i-1b
```

</details>

## USearchMolecules

[USearchMolecules][usm] is a corpus of small molecules with pre-computed binary fingerprints at four widths: MACCS 166 bits, PubChem 881 bits, ECFP4 2048 bits, FCFP4 2048 bits.
Three subsets are hosted on AWS Open Data as Parquet shards: PubChem at 115M molecules, GDB-13 at 977M, and Enamine REAL at 6.04B.
The vectors are genuinely binary rather than quantised floats, a natural fit for Hamming and Jaccard.

`retri-download-molecules` fetches the requested fingerprint column, `--source` picks the subset and `--limit N` takes a prefix of it.
`--fingerprint ecfp4` multiplies the base-file size by roughly 12× at each scale.
Ground-truth time dominates at billion scale, while its memory stays fixed: the base is scanned in tiles, so only one tile and its distances are resident at a time.
The base lands in parts of `--part-rows` rows, 512M by default, as `base.part_000.b1bin` onward, which the `'base.part_*.b1bin'` glob reads as one set.
A rerun into the same folder keeps the finished parts and fetches only the shards after them.
Queries and ground truth are written only with `--top-k`; without it, the parts alone serve self-recall runs, `--self-search`.
Ground truth names neighbors by `u32` ids, so it covers at most 2³² base rows.

<details>
<summary>PubChem 115M MACCS — b1, 168 bits, Hamming, ~2.4 GB</summary>

```sh
cargo install --path . --features download
retri-download-molecules \
    --source pubchem --fingerprint maccs \
    --query-count 10000 --top-k 10 \
    --output data/pubchem-maccs/
retri-eval-usearch \
    --base-vectors 'data/pubchem-maccs/base.part_*.b1bin' \
    --query-vectors data/pubchem-maccs/query.10000.b1bin \
    --query-neighbors data/pubchem-maccs/groundtruth.10000.ibin \
    --data-type b1 --metric hamming,jaccard \
    --output results/pubchem-maccs
```

</details>

<details>
<summary>PubChem 115M ECFP4 — b1, 2048 bits, Hamming, ~29 GB</summary>

```sh
retri-download-molecules \
    --source pubchem --fingerprint ecfp4 \
    --query-count 10000 --top-k 10 \
    --output data/pubchem-ecfp4/
retri-eval-usearch \
    --base-vectors 'data/pubchem-ecfp4/base.part_*.b1bin' \
    --query-vectors data/pubchem-ecfp4/query.10000.b1bin \
    --query-neighbors data/pubchem-ecfp4/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/pubchem-ecfp4
```

</details>

<details>
<summary>GDB-13 977M MACCS — b1, 168 bits, Hamming, ~21 GB</summary>

```sh
retri-download-molecules \
    --source gdb13 --fingerprint maccs \
    --query-count 10000 --top-k 10 \
    --output data/gdb13-maccs/
```

</details>

<details>
<summary>Enamine REAL 6.04B MACCS — b1, 168 bits, Hamming, ~127 GB</summary>

```sh
retri-download-molecules \
    --source enamine --fingerprint maccs \
    --query-count 10000 --top-k 10 \
    --output data/enamine-maccs/
```

</details>

## Cohere Wikipedia Multilingual

[Cohere Wikipedia][cohere-wiki] holds 247M Wikipedia paragraphs embedded with Cohere Embed v3 and bit-packed into 1024-bit `emb_ubinary` columns, 128 bytes per vector.
The dataset also ships the title, paragraph body and URL of every row.
`retri-download-cohere` extracts them into aligned newline-delimited files for semantic-search demos, unless `--no-text` is given.

<details>
<summary>English subset 41.5M — b1, 1024 bits, Hamming, ~5.3 GB</summary>

```sh
retri-download-cohere \
    --language en \
    --query-count 10000 --top-k 10 \
    --output data/cohere-en/
retri-eval-usearch \
    --base-vectors data/cohere-en/base.41488110.b1bin \
    --query-vectors data/cohere-en/query.10000.b1bin \
    --query-neighbors data/cohere-en/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/cohere-en
```

FAISS binary indexes through `IndexBinaryHNSW` also work: pass `--data-type b1`, and the metric is Hamming by construction.

```sh
retri-eval-faiss \
    --base-vectors data/cohere-en/base.41488110.b1bin \
    --query-vectors data/cohere-en/query.10000.b1bin \
    --query-neighbors data/cohere-en/groundtruth.10000.ibin \
    --data-type b1 --metric hamming \
    --output results/cohere-en-faiss
```

</details>

## Unum WikiVerse

[WikiVerse][wikiverse] is a multi-model embedding dataset built on [FineWiki][finewiki]: 61.5M articles across 325 languages, embedded by five models.
Those are Qwen3-Embedding-0.6B at 1024d, GTE-ModernColBERT-v1 at 128d per token, Snowflake arctic-embed-l-v2.0 at 1024d, nomic-embed-text-v1.5 at 768d and e5-mistral-7b-instruct at 4096d.
Each `.f16bin` shard is row-aligned with its source FineWiki parquet, laid out as `<model>/<lang>wiki/<group>_<shard>.{body,title}.f16bin`.
The full corpus is 95-505 GB depending on the model; ColBERT-style embeddings reach 6.2 TB at one vector per token.

Two prerequisites are still pending on the RetriEval side.
`Dataset::load` does not yet recognize the `.f16bin` extension, which shares the `.fbin` header and only swaps `f32` for `f16` in `ScalarFormat`.
The ColBERT model needs the deferred multi-vector plan.
The example below assumes `f16` support has landed.

<details>
<summary>English subset, Qwen3-Embedding-0.6B — f16, 1024d, ~13 GB</summary>

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/unum-cloud/WikiVerse data/wikiverse/
cd data/wikiverse
hf download unum-cloud/WikiVerse \
    --repo-type dataset \
    --include "qwen3-embedding-0.6b/enwiki/*.body.f16bin"
cd ../..
retri-eval-usearch \
    --base-vectors 'data/wikiverse/qwen3-embedding-0.6b/enwiki/*.body.f16bin' \
    --query-vectors data/wikiverse/qwen3-embedding-0.6b/enwiki/000_00000.body.f16bin \
    --data-type f16 --metric cos \
    --output results/wikiverse-en-qwen3
```

The `--base-vectors` glob picks up every English shard in natural-sort order; queries reuse one shard until the official query and ground-truth split lands upstream.

</details>

[unum-wiki-1m]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-arxiv-2m]: https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m
[msft-spacev]: https://github.com/ashvardanian/SpaceV
[msft-turing]: https://big-ann-benchmarks.com/neurips21.html
[yandex-t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[yandex-deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[meta-bigann]: http://corpus-texmex.irisa.fr/
[usm]: https://github.com/unum-science/USearchMolecules
[cohere-wiki]: https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary
[wikiverse]: https://huggingface.co/datasets/unum-cloud/WikiVerse
[finewiki]: https://huggingface.co/datasets/HuggingFaceFW/finewiki
