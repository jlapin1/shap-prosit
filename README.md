# SHAP-Prosit
Python package to extract and visualize SHAP values from Prosit models.

### How to use?
To run whole pipeline for extraction and visualization of SHAP values:
```shell
python -m src <config_path>
```
This will create folder with the same name as ion in config, generate there "output.txt" file with SHAP values and create images with plots of SHAP values.

To run only SHAP values extraction:
```shell
python src/shap_calculator.py <config_path>
```
This will create folder with the same name as ion in config and generate there "output.txt" file with SHAP values.

To run only SHAP values visualization:
```shell
python src/shap_visualization.py <config_path>
```
This will generate plots of SHAP values in the same folder, where given "output.txt" is located.

### Config structure [OUTDATED]

| Variables      | Functionality                                                              | Needed to run  \_\_main\_\_ pipeline | Needed to run  shap_calculator.py | Needed to run  shap_visualization.py |
|----------------|----------------------------------------------------------------------------|:------------------------------------:|:---------------------------------:|:------------------------------------:|
| val_inps_path  | Path to validation dataset that was  generated during model training.      |          :white_check_mark:          |         :white_check_mark:        |                  :x:                 |
| model_path     | Path to trained model.                                                     |          :white_check_mark:          |         :white_check_mark:        |                  :x:                 |
| ion            | Which ion to take as output.                                               |          :white_check_mark:          |         :white_check_mark:        |                  :white_check_mark:                 |
| perm_path      | Path to permutation of sequences.  Can be left empty to generate new.      |          :white_check_mark:          |         :white_check_mark:        |                  :x:                 |
| samp           | Number of times to re-evaluate the model  when explaining each prediction. |          :white_check_mark:          |         :white_check_mark:        |                  :x:                 |
| bgd_sz         | Size of background dataset                                                 |          :white_check_mark:          |         :white_check_mark:        |                  :x:                 |
| sv_path        | Path to SHAP values                                                        |                  :x:                 |                :x:                |          :white_check_mark:          |
| from_which_end | From which end to take AAs  for clustering (right\|left).                  |          :white_check_mark:          |                :x:                |          :white_check_mark:          |
| number_of_aa   | Number of AAs to take for clustering.                                      |          :white_check_mark:          |                :x:                |          :white_check_mark:          |
