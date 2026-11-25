# Changelog

Experiments on USearch's modalities and the baselines they are compared against, one dated section per run day, newest first.
Each section names its hardware once, and every row keeps connectivity 32 and top-10 unless stated.
Recall is judged against an exact oracle over 256 queries unless stated, and the scanned share is distances per query over the base size.
QPS marked † was measured over only 1,000 queries per run, which underfills a B300; later runs use 100,000.

## November 13, 2026

### USearch and FAISS on Graviton4

Both engines on a single-socket AWS Graviton4 host, 84 cores, each matching 64 links per vector: USearch 3.0 at connectivity 64 in its single-layer graph, FAISS 1.15.1 at M=32 in its HNSW.
Both store Deep as `bf16`, and memory is each backend's peak figure.

| Dataset      | Backend      | Type   | Connectivity | Layer 0 | Expansions | 1-r@10 | 10-r@10 |   Add/s | Search/s |  Memory |  Build |
| :----------- | :----------- | :----- | -----------: | ------: | ---------: | -----: | ------: | ------: | -------: | ------: | -----: |
| ECFP4 115.6M | USearch 3.0  | `b1`   |           64 |      64 |  512 / 512 | 0.9931 |  0.9953 |  41,008 |   49,048 | 68.6 GB |  47.0m |
| ECFP4 115.6M | FAISS 1.15.1 | `b1`   |           32 |      64 | 512 / 1024 | 0.9870 |  0.9925 |  27,175 |   11,750 | 62.9 GB | 1h 11m |
| SIFT 100M    | USearch 3.0  | `u8`   |           64 |      64 |  128 / 128 | 0.9921 |  0.9775 | 185,046 |  208,835 | 46.7 GB |   9.0m |
| SIFT 100M    | FAISS 1.15.1 | `u8`   |           32 |      64 |  256 / 128 | 0.9912 |  0.9793 |  41,917 |   93,549 | 42.3 GB |  39.8m |
| SpaceV 100M  | USearch 3.0  | `i8`   |           64 |      64 |  512 / 384 | 0.9934 |  0.9929 |  38,713 |   57,971 | 45.1 GB |  43.1m |
| SpaceV 100M  | FAISS 1.15.1 | `i8`   |           32 |      64 |  512 / 640 | 0.9912 |  0.9908 |  14,975 |   15,773 | 39.5 GB | 1h 51m |
| Deep 100M    | USearch 3.0  | `bf16` |           64 |      64 |  256 / 192 | 0.9934 |  0.9804 |  87,964 |  118,976 | 53.1 GB |  18.9m |
| Deep 100M    | FAISS 1.15.1 | `bf16` |           32 |      64 |  256 / 256 | 0.9929 |  0.9827 |  48,503 |   56,920 | 48.6 GB |  34.4m |

- On ECFP4 the harness sets FAISS's expansions directly, and FAISS still levels off at 0.9870, where it builds faster than it searches.

## November 9, 2026

### USearch and FAISS on Graviton5

Both engines on AWS `m9g.48xlarge`: one Graviton5 socket of 192 Neoverse V3 cores without SMT, two NUMA nodes, 768 GiB, all 192 threads, at connectivity 32.
FAISS's HNSW keeps 64 edges per vector at layer 0, and USearch's single-layer graph keeps 32.
FAISS 1.15.1 is built from source with its SVE kernels and OpenBLAS; USearch 3.0 dispatches its `apple-m2` class there, so its distances run on NEON `sdot` kernels rather than SVE.
Each engine builds once per construction expansion and climbs the search expansion until 1-recall@10 reaches 0.99, keeping the first row that searches at least as fast as it builds.
Inputs and ground truth are SIFT 100M from the BigANN files and PubChem ECFP4 over all 115,627,256 molecules with 10,000 queries; memory is each backend's peak figure.

| Dataset      | Backend      | Type | Expansions | 1-r@10 | 10-r@10 |   Add/s | Search/s |  Memory | Build |
| :----------- | :----------- | :--- | ---------: | -----: | ------: | ------: | -------: | ------: | ----: |
| SIFT 100M    | USearch 3.0  | `u8` |  256 / 160 | 0.9907 |  0.9807 | 292,833 |  184,891 | 33.8 GB |  5.7m |
| SIFT 100M    | USearch 3.0  | `u8` |  512 / 160 | 0.9925 |  0.9824 | 161,952 |  173,823 | 33.8 GB | 10.3m |
| SIFT 100M    | FAISS 1.15.1 | `u8` |  128 / 192 | 0.9927 |  0.9833 | 166,868 |  139,310 | 56.9 GB | 10.0m |
| SIFT 100M    | FAISS 1.15.1 | `u8` |  256 / 128 | 0.9917 |  0.9792 |  84,536 |  178,600 | 56.9 GB | 19.7m |
| ECFP4 115.6M | USearch 3.0  | `b1` |  256 / 512 | 0.9787 |  0.8982 | 203,337 |   55,601 | 53.6 GB |  9.5m |
| ECFP4 115.6M | USearch 3.0  | `b1` |  512 / 512 | 0.9821 |  0.9038 | 104,412 |   57,807 | 53.6 GB | 18.5m |
| ECFP4 115.6M | USearch 3.0  | `b1` | 512 / 1024 | 0.9843 |  0.9073 | 104,412 |   27,404 | 53.6 GB | 18.5m |
| ECFP4 115.6M | FAISS 1.15.1 | `b1` |    40 / 16 | 0.5623 |  0.4704 | 341,635 |  839,862 | 93.3 GB |  5.6m |

- On SIFT the two engines search at parity once both build slower than they search: USearch builds 1.9 times faster, and FAISS answers 3% more queries per second.
- The first row of each SIFT pair reaches 99% but builds faster than it searches, so the second, wider build is the one kept in the README.
- ECFP4 never reaches 99% on the CPU: USearch levels off near 0.984 where the B300 reaches 0.9906 at 512 / 512, and its 10-recall@10 stays near 0.90.
  That gap is tie scoring, since Hamming distances tie often and 10-recall@10 credits only the exact keys the ground truth kept among tied neighbors.
- FAISS's `IndexBinaryHNSW` keeps its compiled-in 40 / 16, since FAISS's parameter tuning has no dispatcher for binary indexes.

## October 22, 2026

### FAISS on CPU

FAISS 1.15.1 from the `faiss-cpu` wheel on a dual-socket Xeon 6768P with all 256 threads, HNSW at connectivity 32.
Each configuration builds once per construction expansion, starting at 128, and climbs the search expansion until 1-recall@10 reaches 0.99.
If that row searches slower than it built, the build repeats at twice the construction expansion; the row kept is the first that searches at least as fast as it built.
Inputs are the BigANN files as published; `u8` rows go through `IndexHNSWSQ` with `QT_8bit_direct`, which stores the bytes exactly, and `f32` rows through `IndexHNSWFlat`.
Memory is the process's peak resident set, the memory-mapped base included.

| Dataset  | Index            | Expansions | 1-r@10 | 10-r@10 |   Add/s | Search/s |  Memory | Build |
| :------- | :--------------- | ---------: | -----: | ------: | ------: | -------: | ------: | ----: |
| SIFT 10M | SQ8 direct, `u8` |   128 / 96 | 0.9900 |  0.9765 | 115,643 |  133,919 |  8.6 GB |  1.4m |
| SIFT 10M | Flat, `f32`      |   128 / 96 | 0.9900 |  0.9765 |  59,688 |   61,027 | 14.4 GB |  2.8m |

- Storing SIFT's bytes as 8-bit codes keeps every neighbor of the `f32` index, since the bytes convert exactly, while building and searching about twice as fast.
- FAISS reaches 99% at a construction expansion of 128 on SIFT 10M, so neither row needed a wider build to search faster than it inserts.

## October 21, 2026

Every run that day used one NVIDIA B300 holding the whole index in VRAM.

### Shapes Chamfer

PubChem conformers: conformer 0 of each embedded molecule, `f16` xyz with hydrogens, about 49.5 atoms per cloud, median 45, p99 140, max 192.
The largest single file holds 85M clouds, since 100M would overflow the `u32` point count of a `.f16bin` header.

| Dataset                | Metric    | Expansion add | Build IPS | Distances / insert | Expansion search | Recall@10 |    QPS | Scanned |
| :--------------------- | :-------- | ------------: | --------: | -----------------: | ---------------: | --------: | -----: | ------: |
| pubchem-conformers-10m | Chamfer   |           256 |     2.50K |             32,755 |               32 |     0.906 | 20.3K† |       … |
| pubchem-conformers-10m | Chamfer   |           256 |     2.50K |             32,755 |              256 |     0.975 |  8.4K† |  0.061% |
| pubchem-conformers-10m | Chamfer   |           128 |     3.71K |             22,002 |               64 |     0.896 | 18.0K† |  0.019% |
| pubchem-conformers-10m | Chamfer   |           128 |     3.71K |             22,002 |              256 |     0.971 |  7.9K† |  0.059% |
| pubchem-conformers-10m | Chamfer   |           128 |     3.71K |             22,002 |              512 |     0.981 |  4.9K† |  0.109% |
| pubchem-conformers-10m | Hausdorff |           256 |     4.30K |                  … |              256 |     0.718 | 12.1K† |       … |
| pubchem-conformers-1m  | Chamfer   |           256 |     2.71K |             31,060 |              256 |     0.970 |  9.3K† |  0.618% |

- Insertion expansion 128 matches 256 in recall for a 48% faster build, so 128 is the better default for conformers.
- Larger insert batches cut distances per insert: at 1M, 256 rows per core gave 31,060 and 2.71K IPS, 1,024 gave 25,477 and 3.29K IPS, 4,096 gave 23,653 and 3.52K IPS, all at recall@10 0.970 to 0.973.
- Hausdorff builds a far less navigable graph than Chamfer on the same clouds.
- Each Chamfer distance scores 8 axis-sign flips in both directions, about 40K point pairs, so a 2.5K IPS build already runs near a third of the B300's FP32 SIMT peak.
- The 256-query CPU oracle over 10M clouds took 950 s.

### USearch on B300

USearch 3.0 with the whole index in VRAM; every row searches at least as fast as it builds, and memory is the backend's peak figure.

| Dataset          | Backend     | Type   | Connectivity | Layer 0 |  Expansions | 1-r@10 | 10-r@10 |   Add/s |  Search/s |  Memory | Build |
| :--------------- | :---------- | :----- | -----------: | ------: | ----------: | -----: | ------: | ------: | --------: | ------: | ----: |
| MACCS 100M       | USearch 3.0 | `b1`   |           32 |      32 |    256 / 64 | 0.9907 |  0.9906 | 559,140 | 1,460,000 | 18.3 GB |  3.4m |
| ECFP4 115.6M     | USearch 3.0 | `b1`   |           32 |      32 |   512 / 512 | 0.9906 |       … | 146,810 |   285,230 |       … | 13.1m |
| SIFT 10M         | USearch 3.0 | `u8`   |           32 |      32 |   256 / 128 | 0.9911 |  0.9780 | 558,510 | 1,040,000 |  2.6 GB |  0.3m |
| SIFT 100M        | USearch 3.0 | `u8`   |           32 |      32 |   256 / 256 | 0.9913 |  0.9810 | 431,520 |   604,490 | 24.8 GB |  3.9m |
| Turing-ANNS 10M  | USearch 3.0 | `bf16` |           32 |      32 | 1024 / 1024 | 0.9908 |  0.9831 |  95,870 |   155,810 |  3.4 GB |  1.7m |
| Turing-ANNS 100M | USearch 3.0 | `bf16` |           32 |      32 | 1024 / 1024 | 0.9933 |  0.9868 |  88,660 |   156,790 | 32.1 GB | 18.8m |

### cuVS on SIFT

cuVS 26.10 CAGRA through NVIDIA's `cuvs-bench` harness rather than `retri-eval-cuvs`, at graph degree 64 and intermediate degree 128 with NN-descent, `search_width` 4, and one thread issuing 10,000-query batches with the queries already in device memory.

| Dataset   | Backend    | Type | Connectivity | Layer 0 | Expansions | 10-r@10 |   Add/s |  Search/s | Build |
| :-------- | :--------- | :--- | -----------: | ------: | ---------: | ------: | ------: | --------: | ----: |
| SIFT 10M  | cuVS 26.10 | `u8` |           64 |      64 |  128 / 128 |  0.9966 | 204,838 | 1,204,000 |  0.8m |
| SIFT 100M | cuVS 26.10 | `u8` |           64 |      64 |  128 / 256 |  0.9963 | 154,799 |   582,000 | 10.8m |

- `cuvs-bench` reports K-recall@10 only, stricter than 1-recall@10; the best of the sweep was 0.9999 at 10M and 0.9993 at 100M.

### Sequence Alignment

HG002 reads are 148-nt Illumina reads of the NIST 300x run: `hg002-reads-*` sampled across the genome at 0.48x coverage, `hg002-chr20-*` all of chromosome 20 shuffled, about 23x coverage at 10M.
Proteins are UniProt Swiss-Prot, 387K unique sequences after holding out queries.
Needleman-Wunsch scores packed class codes through a 32×32 table of `u4` costs with one linear gap: BLOSUM62 folded to costs for proteins, transitions 1, transversions 2 and gap 2 for DNA.

| Dataset               | Metric           | Build IPS | Expansion search | Recall@10 |    QPS | Scanned | Tied top-10 |
| :-------------------- | :--------------- | --------: | ---------------: | --------: | -----: | ------: | ----------: |
| hg002-reads-10m       | Levenshtein      |    18.85K |               32 |     0.381 | 93.2K† |  0.025% |           … |
| hg002-reads-10m       | Levenshtein      |    18.85K |              512 |     0.593 | 22.7K† |  0.125% |           … |
| hg002-reads-10m       | Overlap 8,12     |     3.82K |               32 |     0.560 |   64K† |  0.013% |           … |
| hg002-reads-10m       | Overlap 8,12     |     3.82K |              512 |     0.802 |   14K† |  0.070% |           … |
| hg002-reads-100m      | Levenshtein      |    10.86K |              512 |     0.502 | 18.8K† |  0.014% |           … |
| hg002-chr20-10m       | Levenshtein      |         … |              512 |     0.603 | 25.5K† |  0.127% |           … |
| hg002-chr20-10m       | Overlap 8,12     |         … |              512 |     0.530 | 16.7K† |  0.052% |           … |
| hg002-chr20-10m       | Overlap 4,8,12   |         … |              512 |     0.293 | 19.4K† |  0.020% |           … |
| uniprot-swissprot     | Levenshtein      |       773 |               32 |     0.766 |  15.7K |   0.37% |       46.2% |
| uniprot-swissprot     | Levenshtein      |       773 |              512 |     0.968 |   3.0K |   1.94% |       46.2% |
| uniprot-swissprot     | Needleman-Wunsch |       456 |               32 |     0.851 |   2.6K |   0.35% |       15.2% |
| uniprot-swissprot     | Needleman-Wunsch |       456 |              128 |     0.934 |   1.6K |   0.55% |       15.2% |
| uniprot-swissprot     | Needleman-Wunsch |       456 |              512 |     0.975 |    618 |   1.67% |       15.2% |
| hg002-reads-10m       | Levenshtein      |    15.19K |               32 |     0.397 | 321.8K |  0.022% |       86.8% |
| hg002-reads-10m       | Levenshtein      |    15.19K |              512 |     0.617 |  73.5K |  0.114% |       86.8% |
| hg002-reads-10m       | Overlap 8,16     |     3.94K |               32 |     0.535 | 138.4K |  0.012% |       54.5% |
| hg002-reads-10m       | Overlap 8,16     |     3.94K |              512 |     0.804 |  22.6K |  0.062% |       54.5% |
| hg002-reads-10m       | Needleman-Wunsch |     1.24K |               32 |     0.403 |  25.8K |  0.022% |       78.2% |
| hg002-reads-10m       | Needleman-Wunsch |     1.24K |              128 |     0.503 |  15.4K |  0.037% |       78.2% |
| hg002-reads-10m       | Needleman-Wunsch |     1.24K |              512 |     0.610 |   4.5K |  0.111% |       78.2% |
| hg002-chr20-10m       | Needleman-Wunsch |     1.17K |               32 |     0.595 |  25.1K |  0.023% |       59.4% |
| hg002-chr20-10m       | Needleman-Wunsch |     1.17K |              512 |     0.800 |   4.2K |  0.117% |       59.4% |
| leipzig-latinized-10m | Needleman-Wunsch |     1.91K |               32 |     0.410 |  21.3K |  0.026% |       88.1% |
| leipzig-latinized-10m | Needleman-Wunsch |     1.91K |              512 |     0.782 |   3.9K |  0.141% |       88.1% |

- Needleman-Wunsch with `u4` costs cuts tied top-10 distances from 46% to 15% on proteins and reaches a given recall at about half the expansion of Levenshtein.
- Swiss-Prot is too small to judge either metric: both scan 0.4% to 2% of the base, so the same comparison is queued on 10M UniRef50 clusters.
- The first GPU Needleman-Wunsch kernel is 5 to 6 times slower per query than Levenshtein and builds DNA at 1.2K to 1.7K IPS against 19K, so its speed says nothing about the metric yet.
- Recall on DNA reads falls with scale and with coverage: duplicate reads at 23x coverage form tie basins that neither Levenshtein nor overlap leave.
- Overlap's only knob is its window widths, hashing every window at every width; adding width 4 hurt badly because 4-mers match almost anything in DNA.
- Rows with 100,000 queries hold out the same reads for every metric; on reads Needleman-Wunsch leads Levenshtein by 0.03 to 0.05 at expansions 128 and 256 and ties it at 512, at 12 times the GPU time.
- `leipzig-latinized-10m` is all 24 Leipzig 1M corpora folded through AnyASCII to lowercase a-z, the space, one digit class and one punctuation class, deduplicated, 10M lines; Needleman-Wunsch there uses unit costs, so 88% of its top-10 distances tie.

### Misspelled Text

WikEd Error Corpus, cleaned English: 28,588,505 corrective Wikipedia edit pairs, of which 7,431,814 differ in exactly one token and 1,049,020 keep an old token that is a non-word, under 10 occurrences on the corrected side, within Damerau-Levenshtein 2 of the new one.
Words: 5,486,643 base entries, the 253,934 corrected words plus the Leipzig vocabulary as distractors, and 100,000 misspelled words as queries.
Sentences: 11,031,663 base entries, the corrected sentences plus the Leipzig sentences, and 100,000 misspelled sentences as queries, never inserted.
Every arm but the raw one folds through AnyASCII to the 29-class text alphabet; Needleman-Wunsch takes unit costs, QWERTY adjacency at 1 and other substitutions and gaps at 2, or Kernighan, Church and Gale's 1990 confusion counts as -log2 P(typed | intended) over letters scaled to their 44M-word AP corpus, `u4` costs 9 to 15, indels collapsed to one gap of 9, Spearman 0.985 against the exact costs.
A hit is the intended entry at rank 1 or among the top 10; recall@10 is against each metric's own exact oracle.

| Task  | Metric                    | Build IPS | Expansion search | Rank 1 | Top 10 | Recall@10 |    QPS | Scanned | Tied top-10 |
| :---- | :------------------------ | --------: | ---------------: | -----: | -----: | --------: | -----: | ------: | ----------: |
| Words | Levenshtein UTF-8, raw    |    61.67K |               32 |  0.467 |  0.813 |     0.951 | 699.4K |  0.032% |       94.4% |
| Words | Levenshtein UTF-8, raw    |    61.67K |              512 |  0.488 |  0.853 |     0.984 | 244.8K |  0.181% |       94.4% |
| Words | Levenshtein, folded       |   117.49K |               32 |  0.423 |  0.779 |     0.992 | 929.4K |  0.031% |       94.5% |
| Words | Levenshtein, folded       |   117.49K |              512 |  0.446 |  0.817 |     1.000 | 368.9K |  0.182% |       94.5% |
| Words | Needleman-Wunsch, uniform |     6.88K |               32 |  0.423 |  0.780 |     0.992 | 205.1K |  0.031% |       94.5% |
| Words | Needleman-Wunsch, uniform |     6.88K |              512 |  0.447 |  0.817 |     1.000 |  32.9K |  0.182% |       94.5% |
| Words | Needleman-Wunsch, QWERTY  |     6.56K |               32 |  0.426 |  0.780 |     0.997 | 198.8K |  0.032% |       91.6% |
| Words | Needleman-Wunsch, QWERTY  |     6.56K |              512 |  0.430 |  0.793 |     0.999 |  30.8K |  0.192% |       91.6% |
| Words | Needleman-Wunsch, KCG     |     7.56K |               32 |  0.503 |  0.876 |     0.998 | 216.0K |  0.030% |       75.7% |
| Words | Needleman-Wunsch, KCG     |     7.56K |              512 |  0.506 |  0.880 |     1.000 |  32.4K |  0.188% |       75.7% |

- Confusion-table costs find the intended word at rank 1 for 50% of misspellings against 45% for folded Levenshtein, and within the top 10 for 88% against 82%, already at expansion 32.
- QWERTY adjacency loses to unit costs, since its gaps cost twice an adjacent slip while most misspellings insert, drop or swap letters.
- Unit-cost Needleman-Wunsch matches folded Levenshtein entry for entry, as it should; folding loses about 4 points of rank 1 against the raw UTF-8 arm, from accents and scripts the fold merges.

### Geospatial Relations

Panoramax street-level pictures from the October 11 GeoParquet snapshot: 122.2M kept after dropping placeholder rows, shuffled, with heading in 99% of rows and field of view in 62%.
Terrain is Copernicus GLO-30 over 46°N to 50°N and 3°W to 1°E, 14,400 × 14,400 heights at one arcsecond, lifted to the WGS-84 ellipsoid through EGM2008; 18.7% of the pictures fall on it.
Sight relations use a 100 m radius and unbounded time slack, since each picture records a single instant.

| Dataset        | Relation         | Build IPS | Expansion search | Recall@10 |         QPS | Scanned |
| :------------- | :--------------- | --------: | ---------------: | --------: | ----------: | ------: |
| panoramax-10m  | near             |      521K |               16 |     0.996 |       1.82M | 0.0136% |
| panoramax-10m  | near             |      521K |              256 |     1.000 |       1.54M | 0.0211% |
| panoramax-10m  | visible, terrain |      505K |              256 |     0.986 |       1.36M | 0.0211% |
| panoramax-10m  | sees, terrain    |      505K |              256 |     0.985 |       1.31M | 0.0211% |
| panoramax-10m  | seen, terrain    |      505K |              256 |     0.986 |       1.16M | 0.0211% |
| panoramax-100m | near             |      401K |              256 |     0.999 | 1.35M–2.26M | 0.0024% |
| panoramax-100m | visible, terrain |      394K |              256 |     0.997 | 1.21M–1.90M | 0.0024% |
| panoramax-100m | sees, terrain    |      393K |              256 |     0.992 | 1.19M–1.84M | 0.0024% |
| panoramax-100m | seen, terrain    |      394K |              256 |     0.998 | 1.22M–1.93M | 0.0024% |

- Terrain lines of sight now run on the GPU through NumKong's CUDA terrain walk, and match the CPU's clearances rank for rank on synthetic hills.
- Expansions 16 to 64 compute the same distances, because the GPU walk has a minimum breadth.
- Sight relations lose 1% to 9% recall at low expansion, as blocked or out-of-cone candidates crowd the admitted ones out of the list, and cost 10% to 15% QPS against `near`.
- Pictures without altitude stand on a surface model that includes buildings and trees, so ground-to-ground sight lines are often blocked.

### Kernel Probes

Standalone CUDA probes on a B300, Chamfer pairs per second per SM at the real conformer size distribution.

| Body                                         | Pairs/s/SM | Speedup |
| :------------------------------------------- | ---------: | ------: |
| Warp body, `f32` math                        |       684K |   1.00x |
| Warp body, packed `f32x2` and mixed `f16`    |       823K |   1.20x |
| Same, through inline PTX, shipped body       |       875K |   1.28x |
| Tensor-core block body, one pair per block   |        84K |   0.12x |
| Tensor-core products alone, no drain         |       721K |   1.06x |
| Tensor-core round of 8 to 32 candidates      |        71K |   0.10x |

- Packed `f32x2` arithmetic with the `f16` widening folded into `add.rn.f32.f16` cuts the inner loop from 39.75 to 27 SASS instructions per point, bit-identical on every pair, and gains 1.19x to 1.22x at every size from 16 to 1,024 points.
- The per-pair tensor-core layout is latency-bound at conformer sizes, with 0.41 to 0.89 µs per product round trip; the products alone reach 2.71x to 3.56x only at 128 to 1,024 points.
- Tensor cores cannot meet the exact contract: B300 `mma.sync` aligns every f16 product to the largest and keeps about 25 bits, so every expansion form, even with exactly split f16 halves, misses the error bound on 99.8% to 100% of near-coincident pairs, where SIMT misses none.
- Even inexact, `mma.sync` bodies reach 0.91x on conformers, 1.32x on 300-point Cα clouds and 0.88x on 2,600-atom clouds, against the packed warp body.
- The packed warp body issues about 44% of the SM's lane-instruction slots, and trimming it from 78 to 72 registers adds about 8% through a seventh resident block.
- The one exact formulation that beats SIMT is per-flip branch and bound under a walk radius, at 1.94x to 2.28x on synthetic 300-point Cα clouds, and it needs packs stored in kd order.

## April 20, 2026

### Turing-ANNS on Xeon 6

USearch 2.25 against FAISS 1.12.0, static through `faiss-sys` 0.7.0, on a dual-socket Intel Xeon 6 host with 96 cores and 192 threads, each on its native input type with no rescaling.
Every type runs at the setting that first reaches 99% recall@10 across storage types.

| Dataset          | Backend      | Type   | Connectivity | Layer 0 | Expansions | 1-r@10 | 10-r@10 |  Add/s | Search/s |   Memory |  Build |
| :--------------- | :----------- | :----- | -----------: | ------: | ---------: | -----: | ------: | -----: | -------: | -------: | -----: |
| Turing-ANNS 10M  | USearch 2.25 | `f32`  |           48 |      96 |  768 / 384 | 0.9929 |       … |  8,532 |   12,331 |  13.0 GB |  18.3m |
| Turing-ANNS 10M  | USearch 2.25 | `bf16` |           48 |      96 |  768 / 384 | 0.9929 |       … | 10,496 |   16,940 |  10.9 GB |  14.1m |
| Turing-ANNS 10M  | USearch 2.25 | `f16`  |           48 |      96 |  768 / 384 | 0.9929 |       … | 10,969 |   20,246 |  10.9 GB |  13.5m |
| Turing-ANNS 10M  | USearch 2.25 | `e5m2` |           48 |      96 |  768 / 384 | 0.9919 |       … | 10,526 |   20,534 |   9.8 GB |  13.5m |
| Turing-ANNS 10M  | USearch 2.25 | `e4m3` |           48 |      96 |  768 / 384 | 0.9930 |       … |  7,353 |   12,106 |   9.8 GB |  19.4m |
| Turing-ANNS 10M  | USearch 2.25 | `e3m2` |           48 |      96 |  768 / 384 | 0.9728 |       … | 10,398 |   18,022 |   9.8 GB |  13.3m |
| Turing-ANNS 10M  | USearch 2.25 | `e2m3` |           48 |      96 |  768 / 384 | 0.7941 |       … | 10,935 |   21,313 |   9.8 GB |  13.2m |
| Turing-ANNS 10M  | FAISS 1.12   | `f32`  |           48 |      96 |  768 / 384 | 0.9944 |       … |  7,491 |   16,486 |  14.1 GB |  20.6m |
| Turing-ANNS 10M  | FAISS 1.12   | `bf16` |           48 |      96 |  768 / 384 | 0.9944 |       … |  3,800 |   10,391 |  12.1 GB |  39.4m |
| Turing-ANNS 10M  | FAISS 1.12   | `f16`  |           48 |      96 |  768 / 384 | 0.9944 |       … |  2,545 |   10,032 |  12.1 GB |  1h 1m |
| Turing-ANNS 100M | USearch 2.25 | `f32`  |           48 |      96 |  768 / 384 | 0.9929 |       … |  6,646 |   10,398 | 139.6 GB |  4h 1m |
| Turing-ANNS 100M | USearch 2.25 | `bf16` |           48 |      96 |  768 / 384 | 0.9931 |       … |  8,564 |   14,772 | 105.2 GB |  3h 1m |
| Turing-ANNS 100M | USearch 2.25 | `f16`  |           48 |      96 |  768 / 384 | 0.9930 |       … |  8,807 |   15,412 | 105.2 GB | 2h 54m |
| Turing-ANNS 100M | USearch 2.25 | `e5m2` |           48 |      96 |  768 / 384 | 0.9924 |       … |  7,368 |   13,227 |  88.0 GB | 3h 15m |

## April 18, 2026

### USearch and FAISS on Xeon 6

USearch 2.25 against FAISS 1.12.0, static through `faiss-sys` 0.7.0, on a dual-socket Intel Xeon 6 host with 96 cores and 192 threads, each on its native input type with no rescaling.
MACCS settings are calibrated at 10M and reused at 100M.

| Dataset    | Backend      | Type | Connectivity | Layer 0 | Expansions | 1-r@10 | 10-r@10 |  Add/s | Search/s |  Memory |        Build |
| :--------- | :----------- | :--- | -----------: | ------: | ---------: | -----: | ------: | -----: | -------: | ------: | -----------: |
| MACCS 10M  | USearch 2.25 | `b1` |           32 |      64 |   128 / 64 | 0.9696 |       … | 36,347 |   35,767 |  4.7 GB |         5.0m |
| MACCS 10M  | FAISS 1.12   | `b1` |           64 |     128 |    40 / 16 | 0.9661 |       … | 95,230 |  293,795 |  7.6 GB |         1.5m |
| MACCS 100M | USearch 2.25 | `b1` |           32 |      64 |   128 / 64 | 0.8438 |       … | 35,080 |   38,432 | 40.8 GB |        54.6m |
| MACCS 100M | FAISS 1.12   | `b1` |           64 |     128 |    40 / 16 |      … |       … |      … |        … | ≥ 63 GB | killed at 9h |
| SIFT 10M   | USearch 2.25 | `u8` |           16 |      32 |  128 / 256 | 0.9938 |       … | 35,405 |   80,729 |  4.4 GB |         4.8m |
| SIFT 10M   | FAISS 1.12   | `u8` |           16 |      32 |  128 / 256 | 0.9952 |       … | 26,374 |   38,278 |  5.9 GB |         5.6m |
| SIFT 100M  | USearch 2.25 | `u8` |           16 |      32 |  128 / 256 | 0.9833 |       … | 39,831 |   75,808 | 53.7 GB |        48.7m |
| SIFT 100M  | FAISS 1.12   | `u8` |           16 |      32 |  128 / 256 |      … |       … |      … |        … | ≥ 46 GB | killed at 9h |

- FAISS 1.12 was killed after 9 hours at 100M, past 63 GB on MACCS and 46 GB on SIFT.
- USearch 2.25 builds MACCS 10M faster than it searches.
