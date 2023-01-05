# variational-eqnn-gaussian
An equivariant graph neural network using variational bayesian learning.

## Requirements:
In order to run the scripts within this repository, you will need the graphnn package by Peter Bjørn Jørgensen (pbjo@dtu.dk) et al.
The directory for the repository should look like
* graphnn
* models
* scripts
* tests
* layer_variational.py
* model_evaluation.ipynb
* PAINN_var.py
* runner_variational.py
* var_layer.py

The graphnn folder should then be the folder which contains the entire graphnn project (i.e. another graphnn folder, a scripts folder and more).


## Datasets:
To get the QM9 dataset, use the  graphnn script get_qm9.py.
To get the PC9 dataset, use the get_pc9.py script in scripts

## Pre-trained models
Pretrained variational models trained on the QM9 dataset and the PC9 dataset can be found in the models folder.

## Tests
Precalculated energy estimates for relevant test sets can be found in the tests folder.

For some analysis of these estimates, see the model_evaluation.ipynb.

## Training a new model
To train a new model on a dataset, use the runner_variational.py script.
An example of how to train a variational model on the PC9 dataset is as follows:

In a terminal in the folder for the downloaded repository, use the command

`python3 runner_variational.py --use_painn_model --dataset data/pc9.db --energy_property E --max_steps 3000000 --device cuda --atomwise_normalization --batch_size 64 --initial_lr 0.00007 --forces_weight 0 --output_dir models/PC9_E/model_output_example`
