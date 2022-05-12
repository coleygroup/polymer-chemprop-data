## Diblock Results

The script `prepare-cv-splits.py` is used to create multiple csv files, each with the train/validation/test entries associated with a specific cross-validation fold, as defined by the indices in a pickle file (e.g., `cv10-stratified/cv10-stratified.pkl`). For instance, in the folder `cv10-stratified/master/` you can see csv files called `input_train*.csv`, `input_val*.csv`, and `input_test*.csv`, which contain the data used to train, validate, and test the models across all 10 folds.

The `master` folder contains the results for standard Chemprop. The `polymer-*` folders contain the results obtained with Polymer Chemprop, each tagged with the information that was included in its input representation: stoichiometry (`stoich`), chain architecture (`chain`), polymer size (`Xn`). In each of these folders, the csv file called `dataset-*` contains the full dataset (i.e., not split into train/validation/test).

The script `train_test_rf.py` was used to train the RF models, the results of which are in the folder `cv10-stratified/rf/w_val`. `w_val` indicates that a RF model with default hyperparameters was trained on both the training and validation sets, without hyperparameter tuning (see the published article and the info in the `dataset` folder for details). Each subdirectory of `cv10-stratified/rf/w_val` refers to RF results that used different inputs, as indicated by the name of the respective folders, e.g.:

* `cv10-stratified/rf/w_val/fps_counts` --> fingerprints of monomers as input
* `cv10-stratified/rf/w_val/fps_counts-chain` --> fingerprints with information on chain architecture (via sequence sampling)
* `cv10-stratified/rf/w_val/f1` --> only volume fractions of first block

