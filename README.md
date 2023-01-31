# Perceptual annotation of local distortions in videos: tools and datasets.

This is the README for our software implementation of Maximum Likelihood Difference Scaling to perform inter-content scaling.

## Features

- Maximum Likelihood Difference Scaling for inter-content scaling with Maximum Likelihood Estimation (MLE) and Generalized Linear Model (GLM) implementation in Python3.

- Examples of subjective annotation datasets with intra and inter-content comparisons using pairs, triplets, and quadruplets.

## Installation guidelines

Create an environment with venv in python 3.7 and activate it.

```
$ python3 -m venv .env
$ source .env/Scripts/activate
```

Upgrade pip version if necessary with: 
```
$ pip install --upgrade pip
```

Install the following package with pip manually.
```
$ pip install pandas scipy matplotlib statsmodels==0.12.2 seaborn
```

Or by using the provided requirements.txt file.
```
$ pip install -r requirements.txt
```

## Example of commands to run inter-content estimation


To perform an inter-content scaling for the example dataset named "triplet_dataset.npz" with solver based on MLE and Confidence Interval (CI) estimated with bootstrapping over 100 iterations.
```
$ python demo_estimation_inter_content.py --nb_bootstrap 100 --filename ./datasets/triplet_dataset.npz --solver mle
```

The number of bootstraps can be changed for more or less precision in the estimations. A value of 0 will disable bootstrapping and CI estimation. 

The solver's name can be either "mle" or "glm".

The name of the filename can also be changed between the 3 datasets available in the dataset folder: "pairwise_dataset.npz", "triplet_dataset.npz", and "quad_dataset.npz".

## Example of command to run intra-content estimation

To perform a similar estimation as proposed in the original R implementation of Maximum Likelihood Difference Scaling, it is possible to perform only intra-content scaling.

The following command will estimate scores and generate figures for each tube-content of the datasets.

```
$ python demo_estimation.py --nb_bootstrap 1000 --filename ./datasets/quad_intra.npz --solver mle
```

## Exaample to convert csv file to npz

We provided an example code on how we convert the raw annotations that we obtained in csv files into npz file containing numpy array needed by the solving procedure.

```
$ python convert_datasets.py
```

## Reference paper 

To be defined soon


