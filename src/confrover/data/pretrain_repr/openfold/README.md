# OpenFold representation generation

This module provide utilities to batch generate OpenFold representations for provided sequences. Usage


```bash
# Generate repr to a cache directory
# Run through confrover cli 'confrover openfold_repr' or './make_openfold_repr.py' [...]. See --help for details.
confrover openfold_repr \
    --input_csv </path/to/seqres_index.csv> \ # csv file contains 'seqres' and 'index' columns
    --msa_root </path/to/save/msa> \ # cache directory to save/saved MSA
    --folding_repr </path/to/save/folding_repr> \ # folding representation to generate.
    --num_workers <int>  # number of workers to use.
```
