## Vipea Results

The script `prepare-cv-splits.py` is used to create multiple csv files, each with the train/validation/test entries associated with a specific cross-validation fold, as defined by the indices in a pickle file (e.g., `cv10-random/cv10-random_split.pkl`). For instance, in the folder `cv10-random/master/` you can see csv files called `input_train*.csv`, `input_val*.csv`, and `input_test*.csv`, which contain the data used to train, validate, and test the models across all 10 folds. When `cv9-monomer/cv9-monomer_split.pkl` is used, the corresponding 9 folds are crested instead.

The `master` folder contains the results for standard Chemprop. The `polymer-*` folders contain the results obtained with Polymer Chemprop, each tagged with the information that was included in its input representation for the ablation studie and for the artificial datasets: stoichiometry info only (`only_stoich`), chain architecture only (`chain_only`); artificial dataset with inflated chain importance (`poly_type_x5`), artificial dataset with inflated stoichiometry importance (`comp_x5`). In each of these folders, the csv file called `dataset-*` contains the full dataset (i.e., not split into train/validation/test). The commands used to call Chemprop (and Polymer Chemprop) are in the files `train_test_all_chemprop.sh`.

The script `train_test_rf.py` was used to train the RF models, the results of which are in the folders `cv10-random/rf` and `cv9-monomer/rf`. The `w_val` subdirectory indicates that a RF model with default hyperparameters was trained on both the training and validation sets. The `hopt` subdirectory indicates that the RF models used the validation set for hyperparameter tuning (see the published article and the info in the `dataset` folder for details). The input passed to `train_test_rf.py`, and used to train the RF models, are the pickle files in the folder `datasets/vipea/rf_input/`. With the monomer/polymer representation, and binary/count fingerprints selected according to the folder in which the results are stored. These were, for instance, some the commands used:

```
# from the folder "results/vipea/cv10-random/rf/w_val/monomer-repr/fp_binary"
# cv10, w_val, monomer repr, binary fps
../../../../../train_test_rf.py -f ~/polymer-chemprop-data/datasets/vipea/rf_input/dataset-mono_fps_binary.pkl -k ../../../../cv10-random_split.pkl --train_on_val

# from the folder "results/vipea/cv9-monomer/rf/hopt/polymer-repr/fp_counts"
# cv9, hopt with 20 iterations, polymer repr, count fps
../../../../../train_test_rf.py -f ~/polymer-chemprop-data/datasets/vipea/rf_input/dataset-poly_fps_counts.pkl -k ../../../../cv9-monomer_split.pkl --hopt 20
```

The `data-efficiency` folder contains the scripts and results used to generate Figure S2 in the paper.