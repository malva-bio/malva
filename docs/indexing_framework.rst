Indexing & Query Framework
==========================

.. note::

   This page documents the algorithmic design of the Malva index construction and query
   framework in full detail, enabling re-implementation and reproducibility.
   It corresponds to *Supplementary Note 4* of the Malva manuscript.

Malva implements an **inverted index for the single-cell colored** :math:`k`-**mer indexing
problem**. Given a collection of sequencing libraries :math:`\mathcal{R} = \{R_1, \ldots, R_S\}`
where each library :math:`R_i` contains reads from :math:`n_i` cells, the goal is to support
queries that return, for any :math:`k`-mer :math:`x`, the set of cells

.. math::

   \text{Cells}(x) = \{(i,j) \mid x \text{ appears in cell } c_{ij}\}.

This problem differs from bulk-sample :math:`k`-mer indexing in that the label space grows from
:math:`O(S)` samples to :math:`O(\sum_i n_i)` cells — typically :math:`10^6` to :math:`10^{10}`
unique labels — rendering traditional color-aggregative approaches impractical.

----

Definitions
-----------

The following entities are used and tracked throughout the indexing process:

**Chunk**
   A self-contained index unit comprising multiple samples. Each chunk is stored as an independent
   file and can be queried in parallel.

**Sample**
   A sequencing dataset (e.g., one 10x Chromium run) containing reads from multiple cells. Each
   sample has associated sample-level metadata (e.g., tissue type, donor ID, experimental condition).

**Cell**
   An individual cell within a sample, identified by its barcode. Each cell has associated
   cell-level metadata (e.g., cell type annotation, cluster assignment).

These entities are linked to an external metadata database via composite identifiers, enabling
queries to return not only matching cells but also their biological annotations.

----

Data Structure
--------------

Each chunk contains an inverted index stored in **compressed sparse row (CSR) format** comprising
three arrays:

.. list-table::
   :header-rows: 1
   :widths: 10 15 75

   * - Array
     - Name
     - Description
   * - :math:`\mathbf{I}`
     - ``indices``
     - :math:`N` unique :math:`k`-mers in lexicographically sorted order
   * - :math:`\mathbf{P}`
     - ``indptr``
     - For each :math:`k`-mer at position :math:`i`, :math:`\mathbf{P}[i]` stores the start
       offset into the location array; :math:`\mathbf{P}[i+1] - \mathbf{P}[i]` gives the number
       of cells containing :math:`k`-mer :math:`i`
   * - :math:`\mathbf{D}`
     - ``data``
     - :math:`Z` total cell identifiers (as 32-bit composites), grouped by :math:`k`-mer

This structure supports queries in :math:`O(\log N + L_x)` time where
:math:`L_x = |\text{Cells}(x)|`.

.. code-block:: text

   Sequence array  I:  [ k₀  | k₁  | k₂  | k₃  | ···  | k_{N-1} ]
                             ↓       ↓       ↓
   Pointer array   P:  [  0  |  2  |  5  |  6  |  9  | ···  |  Z  ]
                         ↓              ↓              ↓
   Location array  D:  [ c₀ | c₁ | c₂ | c₃ | c₄ | c₅ | c₆ | c₇ | c₈ | ··· ]

   32-bit cell ID layout:
   ┌─────────────────────┬──────────────────────────┐
   │   sample (s bits)   │   cell (32 − s bits)     │
   └─────────────────────┴──────────────────────────┘

Cell Identifier Encoding
~~~~~~~~~~~~~~~~~~~~~~~~

Cell identifiers stored in :math:`\mathbf{D}` are **32-bit integers** partitioned into two
components:

- **Upper** :math:`s` **bits**: sample ID within the chunk
- **Lower** :math:`32 - s` **bits**: cell ID within the sample

For example, allocating 8 bits for sample ID and 24 bits for cell ID (default) permits up to 256
samples per chunk with approximately 16 million cells each. Alternative allocations (e.g., 9 bits
for sample, 23 bits for cell) allow 512 samples per chunk with approximately 8 million cells each.
The choice depends on the expected cell counts per dataset.

Chunks are kept relatively large (many samples) to improve query efficiency by reducing the number
of separate index files that must be accessed. The chunk ID itself is stored as external metadata
alongside the index file, not within the 32-bit cell identifier.

----

Index Construction
==================

Construction proceeds in three phases:

1. Streaming extraction of :math:`k`-mers from FASTQ files into temporary chunks
2. Merging chunks within each sample
3. Merging samples into multi-sample chunks

K-mer Encoding and Sampling
----------------------------

Each :math:`k`-mer is encoded as a **64-bit unsigned integer** using the standard 2-bit
nucleotide representation (A→00, C→01, G→10, T→11), supporting :math:`k \leq 32`.

From each read of length :math:`\ell`, :math:`k`-mers are extracted at non-overlapping positions
:math:`\{0, k, 2k, \ldots\}` plus a final :math:`k`-mer at position :math:`\ell - k` to ensure
complete coverage. The stride between consecutive :math:`k`-mers equals :math:`k`
(non-overlapping). This sampling reduces storage by a factor of :math:`\sim k` while guaranteeing
that for any query window of length :math:`\geq 2k`, at least one indexed :math:`k`-mer will be
found.

:math:`K`-mers containing ambiguous nucleotides (N) or consisting entirely of a single nucleotide
(homopolymers) are excluded. Chunk-level counts of :math:`k`-mers can be computed from pointer
array offsets. Indexing is **associative** and does not preserve counts per label.

Barcode Handling
~~~~~~~~~~~~~~~~

Cell barcodes are extracted from technology-specific read positions and validated against the
manufacturer's whitelist. Barcodes are encoded as 64-bit integers using the same 2-bit nucleotide
representation as :math:`k`-mers, allowing efficient storage and comparison.

The barcode-to-cell mapping is maintained in an unsorted map associating each encoded barcode with
its integer cell ID (or -1 if absent) within the sample. This structure supports :math:`O(1)`
lookup during read processing. During the final merge phase, local cell IDs are combined with the
sample ID to form the 32-bit composite identifier described above.

Streaming Chunk Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FASTQ files are processed in a streaming fashion. For each read, the barcode is validated and
:math:`k`-mers are extracted. Pairs of :math:`(k\text{-mer},\,\text{cell-id})` are accumulated
in an in-memory buffer until reaching threshold :math:`T` (default :math:`10^8` pairs), at which
point the buffer is flushed to disk as a temporary chunk.

.. admonition:: Algorithm 1: Chunk Flush — Buffer to CSR Arrays

   | **Input:** Buffer :math:`B` of :math:`(k\text{-mer},\ \text{cell-id})` pairs
   | **Output:** Arrays :math:`\mathbf{I}`, :math:`\mathbf{P}`, :math:`\mathbf{D}` written to disk

   1. Sort :math:`B` lexicographically by :math:`(k\text{-mer},\ \text{cell-id})`
   2. Initialize :math:`\mathbf{I} \gets []`, :math:`\mathbf{P} \gets [0]`, :math:`\mathbf{D} \gets []`
   3. Set :math:`(\texttt{prev\_k},\ \texttt{prev\_c}) \gets (\text{null},\ \text{null})`
   4. **For each** :math:`(x, c)` in :math:`B`:

      a. **If** :math:`x \neq \texttt{prev\_k}`:  *(new* :math:`k`-*mer)*

         - Append :math:`x` to :math:`\mathbf{I}`
         - Append :math:`|\mathbf{D}|` to :math:`\mathbf{P}` *(record current position)*
         - Set :math:`\texttt{prev\_c} \gets \text{null}`

      b. **If** :math:`c \neq \texttt{prev\_c}`:  *(deduplicate: presence/absence encoding)*

         - Append :math:`c` to :math:`\mathbf{D}`
         - Set :math:`\texttt{prev\_c} \gets c`

      c. Set :math:`\texttt{prev\_k} \gets x`

   5. Append :math:`|\mathbf{D}|` to :math:`\mathbf{P}` *(final pointer)*
   6. Compress :math:`\mathbf{I}`, :math:`\mathbf{P}`, :math:`\mathbf{D}` blockwise and write to disk

Arrays are compressed blockwise to enable random access. Each array is partitioned into blocks of
:math:`B = 512` elements. For sorted arrays (:math:`\mathbf{I}`, :math:`\mathbf{P}`), delta
encoding may optionally be applied before compression. Compression is performed using the Blosc
meta-compressor framework, which supports multiple underlying codecs including zstd (default), lz4,
and zlib; the choice of codec can be configured based on the desired trade-off between compression
ratio and speed. A chunk table records block offsets for random access.

Chunk Merging
~~~~~~~~~~~~~

After all reads are processed, temporary chunks must be merged. The merge algorithm processes
:math:`k`-mers in lexicographic order using proportional windows from each chunk to bound memory
usage. The window size is set to 5% of each chunk's size (capped at :math:`10^7` entries).

.. admonition:: Algorithm 2: Windowed Multi-Chunk Merge

   | **Input:** Chunks :math:`C_1, \ldots, C_m`, each with arrays :math:`(\mathbf{I}_i, \mathbf{P}_i, \mathbf{D}_i)`
   | **Output:** Merged index :math:`(\mathbf{I}, \mathbf{P}, \mathbf{D})` written to output

   1. Initialize :math:`\texttt{ptr}[i] \gets 0` for all :math:`i`
   2. Set :math:`\texttt{win}[i] \gets \min(0.05 \cdot |\mathbf{I}_i|,\ 10^7)` for all :math:`i`
   3. **While** any chunk has unprocessed entries:

      a. **Find** the maximum :math:`k`-mer safely processable this iteration:

         - Set :math:`k_{\max} \gets \infty`
         - For each active chunk :math:`C_i`: set
           :math:`\texttt{end} \gets \min(\texttt{ptr}[i] + \texttt{win}[i],\ |\mathbf{I}_i| - 1)`,
           then :math:`k_{\max} \gets \min(k_{\max},\ \mathbf{I}_i[\texttt{end}])`

      b. **Collect** all entries :math:`\leq k_{\max}` from all chunks into set :math:`E`:

         - For each chunk :math:`C_i`, while :math:`\mathbf{I}_i[\texttt{ptr}[i]] \leq k_{\max}`:
           add :math:`\bigl(x,\ \mathbf{D}_i[\mathbf{P}_i[\texttt{ptr}[i]]:\mathbf{P}_i[\texttt{ptr}[i]+1]]\bigr)`
           to :math:`E` and advance :math:`\texttt{ptr}[i]`

      c. **Merge** entries sharing the same :math:`k`-mer and write to output:

         - For each unique :math:`x` in :math:`E`: write
           :math:`\bigl(x,\ \text{sorted}(\bigcup\{\texttt{cells} : (x, \texttt{cells}) \in E\})\bigr)`
           to :math:`\mathbf{I}, \mathbf{P}, \mathbf{D}`

   **Key insight:** By taking the *minimum* of the window-end :math:`k`-mers across all chunks, the
   algorithm guarantees that all occurrences of any :math:`k`-mer :math:`\leq k_{\max}` have been
   seen, enabling correct merging without loading entire chunks into memory.

When merging samples into multi-sample chunks, cell IDs are remapped to incorporate the sample
offset within the chunk. The composite identifier then allows downstream queries to recover the
chunk ID, sample ID, and cell ID, which serve as keys into the external metadata database.

Hierarchical Page Index
~~~~~~~~~~~~~~~~~~~~~~~~

For indices where the sequence array exceeds available memory, a hierarchical page structure is
constructed. The sequence array is partitioned into pages of :math:`P = 1024` :math:`k`-mers.
A second-level array samples every :math:`P`-th element; a third level samples every :math:`P`-th
element from the second; and so on until the root level contains fewer than :math:`P` elements.
This structure is stored alongside the main index arrays.

.. code-block:: text

   Level 2 (root):        [         k_max         ]
                           /          |           \
   Level 1:         [k₁₀₂₄]      [k₂₀₄₈]       [···]
                    /      \      /      \
   Base pages:   [P₀]    [P₁]  [P₂]    [P₃]    [P₄]  [···]

   Each base page:    P = 1024 k-mers
   Each level-1 node: max k-mer of P base pages

During lookup, navigation proceeds from root to leaf: at each level, binary search identifies which
child page contains the target :math:`k`-mer, narrowing the search space by factor :math:`P` per
level. For a sequence array of :math:`N` :math:`k`-mers, lookup requires :math:`O(\log_P N)` page
accesses, typically 3–4 for billion-scale indices. The root and intermediate levels are kept in
memory; base pages are loaded on demand.

----

Querying
========

At query time, all chunks are accessed **in parallel**. For each chunk, the query sequence is
decomposed into :math:`k`-mers and matched against the chunk's index. Results from all chunks are
aggregated, and the composite cell identifiers are used to retrieve metadata from the external
database.

Window-Based Matching
---------------------

The query algorithm evaluates sequence similarity using a **windowed voting scheme**. For a query
sequence :math:`Q`, a window of size :math:`w` (default 64 bp) slides with stride 1 (overlapping
windows). Within each window, non-overlapping :math:`k`-mers at positions :math:`\{0, k, 2k,
\ldots\}` are extracted and looked up. A cell matches the window if at least fraction :math:`\tau`
(default 0.65) of the window's :math:`k`-mers are found in that cell. The final score for cell
:math:`c` is the count of windows where :math:`c` achieved a match.

The use of overlapping windows with stride 1 ensures sensitivity to matches at any position within
the query, while the non-overlapping :math:`k`-mer extraction within each window controls
computational cost.

.. admonition:: Algorithm 3: Window-Based Query Scoring

   | **Input:** Query sequence :math:`Q`, index arrays :math:`(\mathbf{I}, \mathbf{P}, \mathbf{D})`, window size :math:`w`, threshold :math:`\tau`
   | **Input:** Abundance filters :math:`f_{\min}` (``count_at_least``), :math:`f_{\max}` (``count_at_most``)
   | **Output:** Map from cell ID to pseudocount score

   1. Initialize :math:`\texttt{scores} \gets \{\}` *(cell → count of matching windows)*
   2. **For** :math:`p \gets 0` **to** :math:`|Q| - w` **step** 1: *(overlapping windows, stride 1)*

      a. :math:`W \gets Q[p : p + w]`
      b. Extract non-overlapping :math:`k`-mers:
         :math:`K \gets \{W[i \cdot k : (i+1) \cdot k] : i = 0, 1, \ldots, \lfloor w/k \rfloor - 1\}`
      c. Filter: :math:`K \gets \{x \in K : x \text{ contains no N}\}`
      d. Initialize :math:`\texttt{votes} \gets \{\}` *(cell → count of* :math:`k`-*mers found)*
      e. **For each** :math:`k`-mer :math:`x \in K`:

         i.   :math:`i \gets \textsc{Lookup}(\mathbf{I}, x)` *(binary search or hierarchical lookup)*
         ii.  **If** :math:`\mathbf{I}[i] \neq x`: skip *(*:math:`k`-*mer not in index)*
         iii. :math:`\texttt{cells} \gets \mathbf{D}[\mathbf{P}[i] : \mathbf{P}[i+1]]`
         iv.  **If** :math:`|\texttt{cells}| < f_{\min}` or :math:`|\texttt{cells}| > f_{\max}`: skip *(filter rare/ubiquitous* :math:`k`-*mers)*
         v.   For each :math:`c \in \texttt{cells}`: increment :math:`\texttt{votes}[c]`

      f. Set :math:`t \gets \lceil \tau \cdot |K| \rceil`
      g. **For each** :math:`(c, v) \in \texttt{votes}`: **if** :math:`v \geq t`, increment :math:`\texttt{scores}[c]`

   3. **Return** :math:`\texttt{scores}`

.. admonition:: Example: :math:`k = 4`,  :math:`w = 8`,  :math:`\tau = 0.65`

   The window yields two :math:`k`-mers (ACGT, ACGA). Each is located in :math:`\mathbf{I}` via
   binary search; pointers in :math:`\mathbf{P}` give the cell range in :math:`\mathbf{D}`.

   - ACGT → cells :math:`\{c_1, c_2\}`
   - ACGA → cells :math:`\{c_1, c_3\}`

   With :math:`\tau = 0.65`, need :math:`t = \lceil 0.65 \times 2 \rceil = 2` matching :math:`k`-mers
   per window.

   - Cell :math:`c_1`: 2/2 :math:`\geq t` → **match** (receives one window vote)
   - Cell :math:`c_2`: 1/2 :math:`< t` → no vote
   - Cell :math:`c_3`: 1/2 :math:`< t` → no vote

Users may specify query-time **abundance filters**:

- ``count_at_least`` (default 0): excludes :math:`k`-mers appearing in fewer than this many cells,
  filtering likely sequencing errors
- ``count_at_most`` (default 100,000): excludes :math:`k`-mers appearing in more than this many
  cells per chunk, filtering repetitive or ubiquitously expressed sequences that might dominate
  runtime and be uninformative

Filters are applied after lookup; no abundance filtering is applied during index construction.

Query Batching and Index Access Pattern
---------------------------------------

For efficiency, all :math:`k`-mers across all query windows are collected, deduplicated, and sorted
lexicographically before lookup. Because the sequence array is sorted, this ensures **sequential
access** through the index, maximizing cache locality and minimizing disk seeks.

The access pattern proceeds as follows:

1. Collect all unique :math:`k`-mers from all query windows
2. Sort :math:`k`-mers lexicographically
3. For each :math:`k`-mer in sorted order:

   a. Binary search (or hierarchical lookup) in sequence array to find position :math:`i`
   b. Read :math:`\texttt{indptr}[i]` and :math:`\texttt{indptr}[i+1]` to determine cell range
   c. Read :math:`\texttt{data}[\texttt{indptr}[i] : \texttt{indptr}[i+1]]` to retrieve cells

4. Redistribute results to originating windows for scoring

For indices exceeding available memory, compressed array blocks are accessed via **memory-mapped
file I/O**. The operating system's virtual memory subsystem handles paging of index blocks on
demand. Decompressed blocks are retained in an LRU cache (capacity :math:`2 \times 10^6` blocks,
approximately 4 GB) to avoid repeated decompression. With sequential access patterns, empirical
cache hit rates exceed 90%.

Metadata Integration
~~~~~~~~~~~~~~~~~~~~

Query results return composite cell identifiers, which are decomposed into (chunk ID, sample ID,
cell ID) tuples serving as keys into a pre-indexed external database. This database stores:

- **Sample-level metadata**: tissue type, donor ID, experimental condition, sequencing platform,
  publication source
- **Cell-level metadata**: cell type annotation, cluster assignment, quality metrics

The separation of sequence index from metadata database allows efficient annotation updates without
rebuilding the :math:`k`-mer index.

----

Parameters Summary
==================

.. list-table:: Default parameter values for index construction and querying
   :header-rows: 1
   :widths: 30 12 15 43

   * - Parameter
     - Symbol
     - Default
     - Description
   * - **Indexing**
     -
     -
     -
   * - K-mer length
     - :math:`k`
     - 24
     - Nucleotides per :math:`k`-mer
   * - Buffer threshold
     - :math:`T`
     - :math:`10^8`
     - Pairs before chunk flush
   * - Block size
     - :math:`B`
     - 512
     - Elements per compressed block
   * - Page size
     - :math:`P`
     - 1024
     - Elements per hierarchical page
   * - Merge window
     - —
     - 5%
     - Fraction of chunk per iteration
   * - Window cap
     - —
     - :math:`10^7`
     - Maximum entries per merge window
   * - Sample ID bits
     - :math:`s`
     - 8
     - Bits for sample within chunk
   * - **Querying**
     -
     -
     -
   * - Query window
     - :math:`w`
     - 64
     - Nucleotides per scoring window
   * - Window stride
     - —
     - 1
     - Overlapping windows
   * - Match threshold
     - :math:`\tau`
     - 0.65
     - Fraction of :math:`k`-mers required per window
   * - Min abundance
     - :math:`f_{\min}`
     - 0
     - ``count_at_least`` filter
   * - Max abundance
     - :math:`f_{\max}`
     - 100,000
     - ``count_at_most`` filter
   * - Cache capacity
     - —
     - :math:`2 \times 10^6`
     - Blocks in LRU decompression cache
