index
***********************

Build a single Malva Index (k-mer index) from single-cell or spatial transcriptomic sequencing reads.

In a single Malva Index, each k-mer is colored by each cell in which it appears. 

.. argparse::
   :filename: ../malva/cli.py
   :func: cmdline_parser
   :prog: malva
   :path: index