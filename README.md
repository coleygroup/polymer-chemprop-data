# A graph representation of molecular ensembles for polymer property prediction

Data and results associated with manuscript ...

Dataset of computed electron affinities and ionization potentials are ...


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
      eprint={},
      archivePrefix={arXiv},
      primaryClass={}
}
```