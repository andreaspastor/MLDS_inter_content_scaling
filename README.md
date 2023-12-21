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

The solver's name can be "mle" or "glm".

The name of the filename can also be changed between the 3 datasets available in the dataset folder: "pairwise_dataset.npz", "triplet_dataset.npz", and "quad_dataset.npz".

## Example of command to run intra-content estimation

To perform a similar estimation as proposed in the original R implementation of Maximum Likelihood Difference Scaling, it is possible to perform only intra-content scaling.

The following command will estimate scores and generate figures for each tube-content of the datasets.

```
$ python demo_estimation.py --nb_bootstrap 1000 --filename ./datasets/quad_intra.npz --solver mle
```

## Example to convert csv file to npz

We provided an example code on how we convert the raw annotations that we obtained in CSV files into ".npz" file containing Numpy array needed by the solving procedure.

```
$ python convert_datasets.py
```

## Active-sampling with AFAD_R algorithm

When running the following command, after the display of the solving, the active-sampling strategy will sample 20 new quadruplets to annotate next by an observer.
```
$ python demo_estimation_inter_content.py --nb_bootstrap 0 --filename ./datasets/quad_dataset.npz --solver mle
```
See comments in the code for detailed explanations.

## Reference papers 

[1] Andréas Pastor and Patrick Le Callet. 2023. Perceptual annotation of local distortions in videos: tools and datasets. In Proceedings of the 14th Conference on ACM Multimedia Systems (MMSys '23). DOI: https://doi.org/10.1145/3587819.3592559

[2] A. Pastor, L. Krasula, X. Zhu, Z. Li and P. Le Callet, "Improving Maximum Likelihood Difference Scaling Method To Measure Inter Content Scale," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 2045-2049, doi: 10.1109/ICASSP43922.2022.9746681.

[3] A. Pastor, L. Krasula, X. Zhu, Z. Li and P. L. Callet, "On the Accuracy of Open Video Quality Metrics for Local Decision in AV1 Video Codec," 2022 IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022, pp. 4013-4017, doi: 10.1109/ICIP46576.2022.9897469.

[4] Andréas Pastor and Patrick Le Callet. 2022. Perception of video quality at a local spatio-temporal horizon: research proposal. In Proceedings of the 13th ACM Multimedia Systems Conference (MMSys '22). Association for Computing Machinery, New York, NY, USA, 378–382. https://doi.org/10.1145/3524273.3533931




