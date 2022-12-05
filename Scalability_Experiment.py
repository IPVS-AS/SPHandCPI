"""
Script for running the experiments regarding the scalability that we described in Section 7 of our paper.
"""

import argparse
import os
import random
import time
import warnings
from itertools import product
from SPH_CPI import SPHandCPI, SPH, RandomForestBorutaMethod, CPI, random_forest_parameters, RandomForestClassMethod

import pandas as pd
import numpy as np

from DataGenerator import ImbalanceGenerator

from Utility import train_test_splitting, update_data_and_training_data, get_train_test_X_y
from Hierarchy import HardCodedHierarchy

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def store_data_to_csv(df_train, df_test, data_output_directory, imb_degree):
    df_train.to_csv(data_output_directory + f"/{imb_degree}_train.csv", index=False)
    df_test.to_csv(data_output_directory + f"/{imb_degree}_test.csv", index=False)


def run_machine_learning(gini_thresholds: list, p_quantile: list, max_info_loss_values: list,
                         imbalance_degree: str = "medium", output_dir: str = ""):


    print("----------------------------------")
    print(f"running with imb = {imbalance_degree}")
    ##############################################################################################
    ################ Setting up output directories based on imbalance degree #####################
    # Default for directories, append the output_directory
    if output_dir == "":
        out_dir = f""
    else:
        out_dir = f"{output_dir}/imbalance_degree/{imbalance_degree}/"

    data_output_directory = f"data/"
    result_output_directory = f"results/"

    if not os.path.exists(data_output_directory):
        os.makedirs(data_output_directory)

    if not os.path.exists(result_output_directory):
        os.makedirs(result_output_directory)
    ##############################################################################################

    result_file = result_output_directory + f"/{imbalance_degree}.csv"
    # Reuse existing results
    if os.path.isfile(result_file):
        acc_result_df = pd.read_csv(result_file, index_col=None)
    else:
        acc_result_df = pd.DataFrame()

    stats_results = pd.DataFrame()
    predictions_results = pd.DataFrame()
    surrogate_results = pd.DataFrame()
    run_id = 1
    n_features = 100
    features_remove_percent = 0.2
    runtime_results = pd.DataFrame()

    for n_samples in n_samples_list:
        # specify number of training samples
        n_train_samples = n_samples * (750 / 1050)
        # missing features in data
        # Random seed for reproducibility
        np.random.seed(run_id * 100)
        random.seed(run_id * 100)

        root_node = HardCodedHierarchy().create_hardcoded_hierarchy()

        generator = ImbalanceGenerator(root=root_node,
                                       imbalance_degree=imbalance_degree,
                                       n_features=n_features,
                                       n_samples_total=n_samples,
                                       features_remove_percent=features_remove_percent)
        data_df = generator.generate_data_with_product_hierarchy()
        root_node = generator.root

        # Train/Test split and update paper_data in the hierarchy
        df_train, df_test = train_test_splitting(data_df, n_train_samples=n_train_samples)
        store_data_to_csv(df_train, df_test, data_output_directory, imbalance_degree)
        root_node = update_data_and_training_data(root_node, df_train, n_features=n_features)
        X_train, X_test, y_train, y_test = get_train_test_X_y(df_train, df_test, n_features=n_features)

        # Dictionary of parameters for the different methods
        methods_to_parameters = {
            RandomForestClassMethod.name(): {"classifier_params": [random_forest_parameters],
                                             "run_id": [run_id]},
            RandomForestBorutaMethod.name(): {"classifier_params": [random_forest_parameters],
                                               "run_id": [run_id]},
            SPH.name(): {"max_info_loss": max_info_loss_values, "hierarchy": [root_node],
                         "run_id": [run_id]},
        SPHandCPI.name(): {"max_info_loss": max_info_loss_values, "hierarchy": [root_node],
                               "gini_threshold": gini_thresholds, "p_threshold": p_quantile,
                               "run_id": [run_id]},
            CPI.name(): {"gini_threshold": gini_thresholds, "p_threshold": p_quantile,
                          "hierarchy": [root_node],
                          "run_id": [run_id]},
        }

        # Run each method in same fashion
        for method in METHODS:
            # Dictionary of parameters to use for each method, retrieve the one for this method
            parameter_dicts = methods_to_parameters[method.name()]

            for parameter_vals in product(*parameter_dicts.values()):

                # 1.) Instantiate method to execute (SPH, SPHandCPI, ...)
                method_instance = method(**dict(zip(parameter_dicts, parameter_vals)))
                print("-------------------------------------------------------------")
                print(f"Running method {method_instance.name()} with n={n_samples}")
                start = time.time()
                # 2.) Fit Method
                method_instance.fit(X_train, y_train)
                fit_time = time.time() - start
                surrogate_sets = 0
                if method_instance.name() == SPH.name() or method_instance.name() == SPHandCPI.name():
                    surrogate_sets = len(method_instance.stats_tracker.surrogate_sets)

                print("---------------------------")
                print(f"Fit time for {method.name()} is: {fit_time}")
                print("---------------------------")

                runtime_df = method_instance.get_runtime_information_df()
                runtime_df["Overall time"] = fit_time
                runtime_df["#surrogates"] = surrogate_sets
                print(runtime_df)

                # 3.) Predict the test samples;
                # No need to use the return value as we use the method_instance object to retrieve
                # the results in a prettier format
                start = time.time()
                method_instance.predict_test_samples(df_test)
                predeiction_time = time.time() - start
                print("---------------------------")
                print(f"prediction time for {method.name()} is: {predeiction_time}")
                print("---------------------------")

                # 4.) Retrieve accuracy Results (A@e and RA@e)
                accuracy_per_e_df = method_instance.get_accuracy_per_e_df()
                print(accuracy_per_e_df)
                avg_a_at_e = accuracy_per_e_df["A@e"].mean()
                ra_at_ten = accuracy_per_e_df[accuracy_per_e_df["R_e"] == 10]["RA@e"].values[0]
                print(avg_a_at_e)
                print(ra_at_ten)

                runtime_df["Avg A@e"] = avg_a_at_e
                runtime_df["A@1"] = accuracy_per_e_df[accuracy_per_e_df["R_e"] == 1]["A@e"].values[0]
                runtime_df["RA@10"] = ra_at_ten
                runtime_df["n"] = n_samples
                runtime_df["prediction time"] = predeiction_time

                runtime_results = pd.concat([runtime_results, runtime_df])
                runtime_results.to_csv("runtime.csv", index=None)


                # Retrieve predicitions
                #predictions = method_instance.get_predictions_df()
                #predictions_results = pd.concat([predictions_results, predictions])
                #predictions_results.to_csv(f"predictions_{imbalance_degree}.csv")

                #stats = method_instance.get_stats_df()
                #stats_results = pd.concat([stats_results, stats])
                #stats_results.to_csv(f"stats_{imbalance_degree}.csv")

                #acc_result_df = pd.concat([acc_result_df, accuracy_per_e_df],
                #                          ignore_index=True)

                print(accuracy_per_e_df)

                #acc_result_df.to_csv(result_file,
                #                     index=False)

if __name__ == '__main__':
    ###############################################################
    ######################## Default Arguments ####################
    # Search space from the paper for the parameters of SPH (max_info_loss) and CPI (gini and p_quantile)

    max_info_loss_values = [
        #0.1,
        #0.15,
         #0.2,
         0.25,
        #0.3,
    #.35,
        #0.4
    ]

    gini_thresholds = [
        # 0.2,
        # 0.25,
        0.3,
        #0.35,
        #0.4
    ]

    p_quantile = [
        #0.7,
        # 0.75,
        0.8,
        #0.85,
        #0.9
    ]
    ###############################################################

    # Use one value of n_samples and n_features but could also use more
    n_samples_list = [
        1050,
        10000,
        100000,
        500000,
        1000000,
        5000000
    ]
    n_features_list = [100]
    features_remove_percent_list = [#0,
                                    0.2
                                    ]

    # Machine learning algorithms to execute
    ###############################################################
    METHODS = [
        #RandomForestBorutaMethod,
        RandomForestClassMethod,
        SPH,
        SPHandCPI,
    ]

    ###############################################################

    imbalance_degree = "medium"

    ###############################################################
    ######### Run Machine Learning  ###############################
    run_machine_learning(
        gini_thresholds=gini_thresholds,
        p_quantile=p_quantile,
        max_info_loss_values=max_info_loss_values,
        imbalance_degree=imbalance_degree,
    )
    ###############################################################
