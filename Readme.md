# Exploiting Domain Knowledge to address Class Imbalance and a Heterogeneous Feature Space in Multi-Class Classification

This Repository contains the prototypical implementations of the submission 
"Exploiting Domain Knowledge to address Class Imbalance and a Heterogeneous Feature Space in Multi-Class Classification".

We provide the code of our classification approach (Section 4 of our paper)
as well as the data generator and the synthetic data we used in the evaluation in Section 7.
However, note that we do not provide the real-world data we used for the evaluation in Section 5 because this data is intellectual property of an industry company that does not allow us to share the data.

## 1. Installation

We require python >= 3.9 for our implementation and also provide a `requirements.txt` listing the required libraries with the specific versions.
These libraries can easily be installed using pip with ``pip install -r requirements.txt``.

## 2. Classification Approach  (Section 4)

In `SPH_CPI.py`, we provide the implementations of the major novel steps for training set preparation of our classification approach (Section 4.1).
These steps are the Segmentation according to Product Hierarchy (SPH, cf. Algorithm 1) and 
the Class Partitioning according to Imbalance (CPI, cf. Algorithm 2).
Note that our python scripts allow to either apply both steps in isolation (see the classes `SPH` and `CPI`) or to apply both sequentially (see class `SPHandCPI`).
For information on how to apply them, we refer to our example notebook `Examples.ipynb`.

## 3. Evaluation with Synthetic Data (Section 7)

For the evaluation in Section 7, we provide the synthetic data, the code to reproduce the experiments,
and the results that we obtained and discussed in our paper.

### 3.1 Data

First, we provide the hierarchy in `Hierarchy.py` that represents the domain knowledge for our synthetic data .
This hierarchy is similar to the product hierarchy from the real-world use case regarding End-of-Line testing that is described in [1] and [2].

We provide the data generator in the script `DataGenerator.py`.
This script allows to generate the synthetic data with varying class distributions as described in our paper (cf. Table 4), i.e., with the 
class distributions "very balanced", "balanced", "medium", "imbalanced", and "very imbalanced".

The synthetic datasets can be found as CSV files in the folder "paper_data/".
We have one sub-folder for each of the five class distributions.
We already separated the data into training and test data, i.e., we provide one CSV file for the training data and one for the test data.

Each dataset contains four categorical features  ("level-0", "level-1, "level-2, and "group), 100 numerical features ("F0", ..., "F99"),
and a "target" column that represents the class label.
For further details, we refer to our example Notebook ``Examples.ipynb``.

### 3.2 Experiments and Results

To run the experiments, we also provide the script `Experiment.py`.
Here, we use the same parameters for SPH, CPI, SPH+CPI, and the RF+B baseline as described in our paper.
You can simply run the experiments using `python Experiment.py`.

However, this might take some time as multiple approaches and different parameterizations are executed.
Hence, we also prepare the results that we discuss in our evaluation, in particular for Sections 7.3 and 7.4, in the folder
"paper_results/".
Here, we provide one CSV file for each dataset.
Each CSV file contains the results for all approaches that we applied.
As we performed a grid search to obtain the best results regarding the parameters of SPH and CPI, we 
provide all results for the different parameterizations. In Section 7.3 and 7.4, we only discuss the results of the most important parameterizations. 

Each CSV file has the following columns:

- `Method`: Name of the used approach. The values are "SPH", "CPI", "SPH+CPI", and "RF+B", respectively.
- `R_e`: Describes the position of the recommendation list. Values are between 1 and 10.
- `A@e`: Expresses the accuracy that is achieved with a recommendation list R_e of length e.
- `RA@e`: Shows the number of rework attempts that are required with a recommendation list R_e of length e.
- `max info loss`: Parameter threshold of SPH. This value is only set if the "Method" is "SPH" or "SPH+CPI", 
otherwise the value is NaN.
- `gini Threshold`: Threshold parameter for the Gini coefficient of CPI. This value is only set if the "Method" is "CPI" or "SPH+CPI", 
otherwise the value is NaN.
- `p value`: Parameter p for p-quantile of CPI. This value is only set if the "Method" is "CPI" or "SPH+CPI", 
otherwise the value is NaN.

  
Note that when you run the experiments in `Experiment.py`, the results are stored in a different folder, i.e., instead of 
"paper_data/" and "paper_results/" the folders are "data/" and "results/".
We do this to prevent overriding the provided paper results accidentally.


## References
[1] Hirsch, Vitali, Peter Reimann, and Bernhard Mitschang. "Exploiting Domain Knowledge to Address Multi-Class Imbalance and a Heterogeneous Feature Space in classification Tasks for Manufacturing Data". Proceedings of the VLDB Endowment 13.12 (2020): 3258-3271, doi: https://doi.org/10.14778/3415478.3415549

[2] V. Hirsch, P. Reimann and B. Mitschang, "Data-Driven Fault Diagnosis in End-of-Line Testing of Complex Products". 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2019, pp. 492-503, doi: https://doi.org/10.1109/DSAA.2019.00064.
