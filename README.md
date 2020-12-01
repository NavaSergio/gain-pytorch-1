# GAIN (Generative Adversarial Imputation Networks) PyTorch

GAIN[1] is an architecture for missing data imputation, using the Generative Adversarial Networks methodology. Here is a PyTorch implementation of the GAIN architecture. Original implementation can be found in : https://github.com/jsyoon0823/GAIN<br>


## Example Runs:

### Vanilla GAIN on SPAM:
`python train_gain.py --data-type spam --miss-rate 0.2 --batch-size 128 --hint-rate 0.9 --alpha 100 --eval-freq 20 --learning-rate 0.001 --max-epochs 200`


### General comments:
To run on GPU, add `--device cuda` as an argument to the scripts. By default, it will run on CPU. <br>


## References:
[1] Jinsung Yoon, James Jordon, Mihaela van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets", International Conference on Machine Learning (ICML), 2018. <br>