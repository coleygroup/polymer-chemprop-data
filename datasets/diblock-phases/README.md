The file `diblock.csv` contains the information on diblock copolymers and their phases.  `names_dict.csv` and `block_info.csv` contain additional information on the (co)polymers forming the blocks, needed to create the inputs for the MPNN and RF models. These data in these files was processed with `parse_dataset.ipynb` to create the inputs for the models tested.

The folder `chemprop_inputs` contains the `csv` files used as input for Chemprop (both standard Chemprop, `dataset-master_chemprop.csv`, and the modified Polymer Chemprop, `dataset-polymer_chemprop*`). The files for Polymer Chemprop further specify the information provided in the input: stoichiometry (`stoich` tag), chain architecture (`chain`), polymer size (`Xn`).

The folder `rf_inputs` contains pickle files with the input used by RF models. Those based on fingerprints have `fps_counts` in their name. Like for the Chemprop inputs, the name specifies the information provided in the input.

The pickle file `cv10-stratified.pkl` contains the indices of the random 10-fold cross validation splits used to train, validate, and test the models.