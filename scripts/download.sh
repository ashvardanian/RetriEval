#!/bin/sh
# Fetches a public dataset into data/<name>/ under the file names the README's commands use.
# Usage: scripts/download.sh <name>, where <name> is one of the `case` arms below.
set -eu
cd "$(dirname "$0")/.."

bigann=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
turing=https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS
yandex=https://storage.yandexcloud.net/yandex-research/ann-datasets
hugging=https://huggingface.co/datasets/unum-cloud

fetch() { # fetch <url> <file>, resuming a partial download
    mkdir -p "data/$name"
    curl -fL --retry 5 -C - -o "data/$name/$2" "$1"
}

slice() { # slice <url> <rows> <bytes-per-row> <output>: a leading prefix of a large file, with its row count patched
    mkdir -p "data/$name"
    curl -fL --retry 5 -r "0-$((8 + $2 * $3 - 1))" -o "data/$name/$4" "$1"
    printf "$(printf '\\%03o' $(($2 & 255)) $(($2 >> 8 & 255)) $(($2 >> 16 & 255)) $(($2 >> 24)))" |
        dd of="data/$name/$4" bs=1 count=4 conv=notrunc status=none
}

name=${1:?usage: scripts/download.sh <name>}
case "$name" in
wiki-1m)
    fetch "$hugging/ann-wiki-1m/resolve/main/base.1M.fbin" base.1M.fbin
    fetch "$hugging/ann-wiki-1m/resolve/main/query.public.100K.fbin" query.public.100K.fbin
    fetch "$hugging/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin" groundtruth.public.100K.ibin
    ;;
arxiv-2m)
    fetch "$hugging/ann-arxiv-2m/resolve/main/abstract.e5-base-v2.fbin" abstract.e5-base-v2.fbin
    fetch "$hugging/ann-arxiv-2m/resolve/main/query.fbin" query.fbin
    fetch "$hugging/ann-arxiv-2m/resolve/main/groundtruth.ibin" groundtruth.ibin
    ;;
sift-10m | sift-100m)
    case "$name" in sift-10m) rows=10000000 scale=10M ;; *) rows=100000000 scale=100M ;; esac
    fetch "$bigann/bigann/query.public.10K.u8bin" query.public.10K.u8bin
    fetch "$bigann/GT_$scale/bigann-$scale" groundtruth.public.10K.ibin
    slice "$bigann/bigann/base.1B.u8bin" "$rows" 128 "base.$scale.u8bin"
    ;;
sift-1b)
    fetch "$bigann/bigann/base.1B.u8bin" base.1B.u8bin
    fetch "$bigann/bigann/query.public.10K.u8bin" query.public.10K.u8bin
    fetch "$bigann/bigann/GT.public.1B.ibin" groundtruth.public.10K.ibin
    ;;
turing-1m | turing-10m | turing-100m)
    case "$name" in turing-1m) rows=1000000 scale=1M ;; turing-10m) rows=10000000 scale=10M ;; *) rows=100000000 scale=100M ;; esac
    fetch "$turing/query100K.fbin" query.public.100K.fbin
    fetch "$turing/msturing-gt-$scale" groundtruth.public.100K.ibin
    slice "$turing/base1b.fbin" "$rows" 400 "base.$scale.fbin"
    ;;
spacev-100m)
    fetch "$hugging/ann-spacev-100m/resolve/main/base.100M.i8bin" base.100M.i8bin
    fetch "$hugging/ann-spacev-100m/resolve/main/query.30K.i8bin" query.30K.i8bin
    fetch "$hugging/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin" groundtruth.30K.i32bin
    ;;
deep-1b)
    fetch "$yandex/DEEP/base.1B.fbin" base.1B.fbin
    fetch "$yandex/DEEP/query.public.10K.fbin" query.public.10K.fbin
    fetch "$yandex/DEEP/groundtruth.public.10K.ibin" groundtruth.public.10K.ibin
    ;;
t2i-1m | t2i-1b)
    case "$name" in t2i-1m) base=base.1M.fbin ;; *) base=base.1B.fbin ;; esac
    fetch "$yandex/T2I/$base" "$base"
    fetch "$yandex/T2I/query.public.100K.fbin" query.public.100K.fbin
    fetch "$yandex/T2I/groundtruth.public.100K.ibin" groundtruth.public.100K.ibin
    ;;
*)
    echo "scripts/download.sh: unknown dataset '$name'" >&2
    exit 1
    ;;
esac
