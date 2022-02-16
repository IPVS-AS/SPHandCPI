"""
Script for running the experiments that we described in Section 7 of our paper.
"""

import argparse
import os
import random
import warnings
from itertools import product
from SPH_CPI import SPHandCPI, SPH, RandomForestBorutaMethod, CPI, random_forest_parameters

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


def run_machine_learning(gini_thresholds: list, p_quantile: list, max_info_loss_values: list, total_runs: int,
                         imbalance_degree: str = "medium", output_dir: str = "",
                         features_remove_percent_list=[0.2]):


    if imbalance_degree == 'all':
        imbalance_degrees = ImbalanceGenerator.imbalance_degrees
    else:
        imbalance_degrees = [imbalance_degree]
    runs = range(1, total_runs + 1, 1)
    ###############################################################

    print(n_features_list)

    for imbalance_degree in imbalance_degrees:

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

        for n_samples in n_samples_list:
            # specify number of training samples
            n_train_samples = n_samples * (750 / 1050)
            for n_features in n_features_list:
                # missing features in data
                for features_remove_percent in features_remove_percent_list:
                    for run_id in runs:
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

                                # 2.) Fit Method
                                method_instance.fit(X_train, y_train)

                                # 3.) Predict the test samples;
                                # No need to use the return value as we use the method_instance object to retrieve
                                # the results in a prettier format
                                method_instance.predict_test_samples(df_test)

                                # 4.) Retrieve accuracy Results (A@e and RA@e)
                                accuracy_per_e_df = method_instance.get_accuracy_per_e_df()
                                print(accuracy_per_e_df)

                                acc_result_df = pd.concat([acc_result_df, accuracy_per_e_df],
                                                          ignore_index=True)

                                print(accuracy_per_e_df)

                                acc_result_df.to_csv(result_file,
                                                     index=False)

if __name__ == '__main__':
    ###############################################################
    ######################## Default Arguments ####################
    # Search space from the paper for the parameters of SPH (max_info_loss) and CPI (gini and p_quantile)

    max_info_loss_values = [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
        0.4]

    gini_thresholds = [
        0.2,
        0.25,
        0.3,
        0.35,
        0.4
    ]

    p_quantile = [
        0.7,
        0.75,
        0.8,
        0.85,
        0.9
    ]
    ###############################################################

    # Use one value of n_samples and n_features but could also use more
    n_samples_list = [1050]
    n_features_list = [100]
    features_remove_percent_list = [0.2]

    # Machine learning algorithms to execute
    ###############################################################
    METHODS = [
        SPH,
        RandomForestBorutaMethod,
        SPHandCPI,
        CPI,
    ]

    ###############################################################
    ######################## Parse Arguments from CMD##############
    parser = argparse.ArgumentParser()
    parser.add_argument("-imbalance",
                        help="Degree of imbalance. This should either be 'very_balanced', 'balanced', 'medium', "
                             "'imbalanced', or 'very_imbalanced'.",
                        default='all', choices=ImbalanceGenerator.imbalance_degrees + ['all'])
    parser.add_argument('-info_loss', type=float,
                        help="Percentage of information loss to use. Default is 25 percent",
                        nargs='*',
                        required=False, default=max_info_loss_values)

    parser.add_argument('-gini', type=float,
                        help='Percentage of the threshold for the gini index. Per default, multiple values from 25 '
                             'to 40 in 5th steps are executed.',
                        nargs='*',
                        required=False, default=gini_thresholds)

    parser.add_argument('-p', type=float,
                        help='Percentage of the thresholds for the p_quantile. Per default, multiple values from 70 '
                             'to 95 in 5th steps are executed',
                        nargs='*',
                        required=False, default=p_quantile)

    parser.add_argument('-runs', type=int,
                        help='Number of runs to perform. The runs differ in different seed values.',
                        required=False, default=1)

    parser.add_argument('-samples', type=int, nargs='*',
                        help='Number of samples to generate with the paper_data generation.',
                        required=False, default=n_samples_list)
    parser.add_argument('-features', type=int, nargs='*',
                        help='Number of samples to generate with the paper_data generation.',
                        required=False, default=n_features_list)
    parser.add_argument('-methods', type=str, nargs='*',
                        help="Methods to execute (SPH, CPI, SPHandCPI, RF, RF+B).",
                        default=METHODS)
    parser.add_argument('-output_dir', type=str,
                        help="Name of the output directory where the results will be stored.",
                        default="")
    parser.add_argument('-missing_features', default=features_remove_percent_list, type=float,
                        required=False,
                        nargs='*',
                        help="Fraction (0 to 1 ) of features to remove. The missing features percentage will be "
                             "in this are but will be a bit higher (~5%).")

    args = parser.parse_args()

    max_info_loss_values = args.info_loss

    gini_thresholds = args.gini

    p_quantile = args.p

    n_samples_list = args.samples
    n_features_list = args.features

    features_remove_percent = args.missing_features
    out_dir = args.output_dir
    imbalance_degree = args.imbalance
    total_runs = args.runs

    if imbalance_degree == 'all':
        imbalance_degrees = ImbalanceGenerator.imbalance_degrees
    else:
        imbalance_degrees = [imbalance_degree]
    runs = range(1, total_runs + 1, 1)
    ###############################################################

    ###############################################################
    ######### Run Machine Learning  ###############################
    run_machine_learning(
        # new, instead of cm_vals:
        gini_thresholds=gini_thresholds,
        p_quantile=p_quantile,
        max_info_loss_values=max_info_loss_values,
        total_runs=total_runs,
        imbalance_degree=imbalance_degree,
        output_dir=out_dir,
        features_remove_percent_list=features_remove_percent
    )
    ###############################################################
