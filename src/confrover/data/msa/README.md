# MSA Module

This module provides utilities to batch query MSA for provided sequences using ColabFold's MSA server. Usage

```bash
# Query MSA and save to a MSA cache directory
# Run through confrover cli 'confrover query_msa' or './mmseq2_colab.py' [...]. See --help for details.
confrover query_msa \
    --input_csv </path/to/seqres_index.csv> \ # csv file contains 'seqres' and 'index' columns
    --msa_root </path/to/save/msa> \
    --max_query_size <int> \ # maximum number of sequences for each query to ColabFold's server.
```
