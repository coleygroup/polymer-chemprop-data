# A graph representation of molecular ensembles for polymer property prediction

Data and results associated with the manuscript [Aldeghi & Coley (2022) "A graph representation of molecular ensembles for polymer property prediction"](https://arxiv.org/abs/2205.08619).

The dataset of computed electron affinities and ionization potentials can be found in the folder `datasets/vipea`, but is also provided as a `csv` as part of the Supporting Information. If you'd like to reproduce the calculations, take a look at the files in this GitHub repo, but if you just want the dataset, you can download it from the [Electronic SI](#).


## Requirements
To parse all data and recreate the input files you need:
* `rdkit >= 2021`
* `natsort`

To reproduce the results:
* `chemprop == 1.4.0-polymer`
* `scikit-learn`

Note that the modified version of Chemprop with weighted edges (wD-MPNN, or "Polymer Chemprop") is needed. This  can be found in the [polymer-chemprop](https://github.com/coleygroup/polymer-chemprop) repository. However, to run only the baselines with the [standard Chemprop](https://github.com/chemprop/chemprop) implementation (D-MPNN), `chemprop >= 1.3.1` is sufficient.

To reproduce the figures:
* `matplotlib`
* `seaborn`

The details of the Python environment that was used for this work can be found in the `environment.yml` file.

## Citation
If you use the dataset or the wD-MPNN version of Chemprop for polymer property prediction please cite

```
@misc{wdmpnn,
      title={A graph representation of molecular ensembles for polymer property prediction}, 
      author={Matteo Aldeghi and Connor W. Coley},
      year={2022},
      eprint={2205.08619},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
