""""
This script contains the implementations of our classification approach and of the baseline.
To this end, we first use an abstract ClassificationMethod that is inherited by all implemented classification approaches.
We provide the implementations of SPH and CPI as standalone approaches, but also the approach to use both sequentially, i.e., SPHandCPI.
Further, this script also contains the implementation of the baseline Random Forest with Boruta (RF+B).
"""

import logging
import time
import warnings
from abc import abstractmethod
from collections import Counter

from anytree import PreOrderIter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from sklearn.pipeline import Pipeline
import concentrationMetrics as cm
import pandas as pd

from Hierarchy import HardCodedHierarchy

random_forest_parameters = {'random_state': 1234,
                            'n_estimators': 50,
                            'verbose': 100,
                            'max_depth': 10,
                            # 'n_jobs': -1
                            }

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def gini(x):
    my_index = cm.Index()
    class_frequencies = np.array(list(Counter(x).values()))
    return my_index.gini(class_frequencies)


class StatisticTracker:
    """
    Tracks statistic about missing feautres, n_features, samples, and the gini index for each partition.
    It is also used for getting the predictions, accuracy and rework attempts for each method.
    """

    def __init__(self):
        self.missing_features = 0
        self.features = 0
        self.classes = 0
        self.samples = 0
        self.gini_index = 0

        self.sph_count = 0
        self.cpi_count = 0

        self.sph_time = 0
        self.cpi_time = 0
        self.model_training_time = 0
        self.boruta_time = 0
        self.prediction_time = 0
        self.knn_training_time = 0

        # keep track of predictions, i.e., which sample with its group and class was predicted correctly at which position
        self.predictions = []

        # keep track of surrogate sets for SPH/ SPH+CPI
        self.surrogate_sets = []

    def set_sph_time(self, sph_time):
        self.sph_time = sph_time

    def add_model_training_time(self, model_time):
        self.model_training_time += model_time

    def add_boruta_training_time(self, boruta_time):
        self.boruta_time += boruta_time

    def add_cpi_time(self, cpi_time):
        self.cpi_time += cpi_time

    def track(self, partitions_for_node, partition):
        if not partition:
            # if not a partitioning approach, we model it as one partition
            partitions_for_node = {"node": {"data": partitions_for_node[0], "labels": partitions_for_node[1]}}

        # count number of partitions --> if we have a tuple we count the length of it!
        # n_partitions = sum([len(x) if isinstance(x, tuple) else 1 for x in partitions_for_node.values()])
        n_partitions = 0

        for partition_key, partition in partitions_for_node.items():

            # make partition iterable (list) so we can iterate over it if we have more partitions
            if not isinstance(partition, tuple):
                partitions = [partition]
            else:
                partitions = partition

            avg_vm = 0
            for part in partitions:
                n_partitions += 1
                data = part["data"]
                labels = part["labels"]

                if -1 in labels:
                    # there was OvA Binarization, do not use that for statistics
                    # filter out -1 from labels and data by storing the indices
                    indices = [ind for ind, label in enumerate(labels) if label != -1]

                    data = np.take(data, indices, axis=0)
                    labels = np.take(labels, indices, axis=0)

                self.classes += len(np.unique(labels))
                gini_index = gini(labels)
                self.gini_index += gini_index

                df = pd.DataFrame(data=data, columns=[f"F{i}" for i in range(data.shape[1])])
                df = df.dropna(axis=1, how='all')

                missing = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
                self.missing_features += missing
                self.samples += data.shape[0]
                self.features += data.shape[1]

        self.classes = self.classes / n_partitions
        self.gini_index = self.gini_index / n_partitions
        self.missing_features = self.missing_features / n_partitions
        self.samples = self.samples / n_partitions
        self.features = self.features / n_partitions

    def get_stats_df(self):
        return pd.DataFrame(data={
            "#Samples": [self.samples],
            "missing": [self.missing_features],
            "#Features": [self.features],
            "#Classes": [self.classes],
            "Gini": [self.gini_index],
            "#SPH": [self.sph_count],
            "#CPI": [self.cpi_count],
        })

    def add_predictions(self, y_probas, y_true, e, groups, class_labels):
        """
        Adds the current predictions to the prediction list.
        To this end, it adds an entry to a dictionary with the group and class of the sample
        and at which position it was predicted correctly among the top-e predictions in y_pred.

        :param y_probas: List of the predictions (result of predict_proba).
        :param y_true: List of actual class values for the predictions
        :param e: Length of recommendation list
        :param groups: list that contains the group name in the same order as y_probas, y_true, i.e.,
         for each sample the belonging group name.
        """
        # get indices of highest probabilities
        best_e = np.argsort(y_probas)[:, :-e - 1:-1]
        # top_e classes in descending order by their probability
        best_e = class_labels[best_e]

        for top_e_pred, y, group in zip(best_e, y_true, groups):

            # check for first occurence that the prediction equals the true y value
            correct_position = np.where(top_e_pred == y)[0]

            # if prediction is not amongst top e, then the correct prediction is 0 (not found)
            if len(correct_position) == 0:
                correct_position = 0
            else:
                # have to add 1, because first correct position gives index 0
                correct_position = correct_position[0] + 1
            self.predictions.append({"group": group, "target": y, "correct_position": correct_position})

    def get_predictions_dict(self):
        return self.predictions

    def get_predictions_df(self):
        return pd.DataFrame(data=self.predictions)

    def add_surrogate_set(self, node_id):
        self.surrogate_sets.append(node_id)

    def get_surrogates_df(self):
        return pd.DataFrame({"surrogate": self.surrogate_sets})

    def set_prediction_time(self, prediction_time):
        self.prediction_time = prediction_time

    def add_knn_time(self, knn_time):
        self.knn_training_time += knn_time


class ClassificationMethod:
    def __init__(self, classifier=RandomForestClassifier,
                 classifier_params=random_forest_parameters,
                 partitioning=False, hierarchy_required=False, hierarchy=None, run_id=1):
        self.classifier = classifier(**classifier_params)
        self.partitioning = partitioning
        self.hierarchy_required = hierarchy_required
        self.hierarchy = hierarchy
        self.run_id = run_id

        # parameters such as p-quantile or gini
        self.parameters = {}

        # Used to track statistics about the resulting partitions, only for methods that do a partitioning
        self.stats_tracker = StatisticTracker()
        self.partitions_for_node = None

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @staticmethod
    def name():
        pass

    @abstractmethod
    def predict_test_samples(self, df_test, e=10):
        """
        Should implement for each method how to predict the test samples.
        For partitioning methods, this means to predict the correct group (assuming there is a "group" column in df_test).
        :param df_test: Dataframe that contains the test test.
        :param e: Length that limits the recommendation list
        :return: The predictions, but should not be required as they can be retrieved with
        get_accuracy_per_e_df() or get_predictions_df().
        """
        pass

    def track_stats(self):
        self.stats_tracker.track(self.partitions_for_node, partition=self.partitioning)

    def get_accuracy_per_e_df(self):
        """
        Returns a dataframe that contains the A@e and RA@e scores for this method.
        TO this end, it requires to already be fitted and made predictions.
        :return: DataFrame that contains the A@e, RA@e scores as well as the method name, the run_id and
        the parameter values
        """
        predictions_df = self.stats_tracker.get_predictions_df()
        accuracies = []

        # get maximum length of recommendation list
        max_e = predictions_df["correct_position"].max()
        for e in range(1, max_e + 1):
            correct_predicted = predictions_df[
                (predictions_df["correct_position"] > 0) & (predictions_df["correct_position"] <= e)]

            # A@e is simply the number of correct predicted for position<=e
            a_at_e = len(correct_predicted) / len(predictions_df)
            # RA@e is simply the sum of all correct predicted positions divided by the amount of correct predictions
            ra_at_e = sum(correct_predicted["correct_position"]) / len(correct_predicted)

            acc_per_e = {"Method": self.name(), "R_e": e, "A@e": a_at_e, "RA@e": ra_at_e, "Run": self.run_id}
            # add parameters for easier identification
            for parameter_key, parameter_value in self.parameters.items():
                acc_per_e[parameter_key] = parameter_value

            accuracies.append(acc_per_e)
        return pd.DataFrame(accuracies)

    def get_predictions_df(self):
        predictions_df = self.stats_tracker.get_predictions_df()
        predictions_df = self._add_parameters_method_run(predictions_df)
        return predictions_df

    def get_surrogates_df(self):
        surrogates_df = self.stats_tracker.get_surrogates_df()
        surrogates_df = self._add_parameters_method_run(surrogates_df)
        return surrogates_df

    def get_stats_df(self):
        stats_df = self.stats_tracker.get_stats_df()
        stats_df = self._add_parameters_method_run(stats_df)
        return stats_df

    def _add_parameters_method_run(self, df):
        """
        Adds parameters, method_name and the run_id for this method to any Dataframe.
        Typically used as wrapper for statistics, predictions etc.

        :param df: dataframe to append the values
        :param run_id: id of the current run
        :return: df with the parameters, Method and run_id as new columns and fixed values
        """
        for parameter_key, parameter_value in self.parameters.items():
            df[parameter_key] = parameter_value
        df["Method"] = self.name()
        df["Run"] = self.run_id
        return df

    def get_runtime_information_df(self):
        runtime_info = {"Method": [self.name()],
                        "SPH time": [self.stats_tracker.sph_time],
                        "CPI time": [self.stats_tracker.cpi_time],
                        "KNN training time": [self.stats_tracker.knn_training_time],
                        "Model Training time": [self.stats_tracker.model_training_time],
                        # "Boruta time": [self.stats_tracker.boruta_time]
                        }
        runtime_df = pd.DataFrame(data=runtime_info)
        self._add_parameters_method_run(runtime_df)
        return runtime_df


class RandomForestClassMethod(ClassificationMethod):
    def __init__(self, classifier=RandomForestClassifier,
                 classifier_params=random_forest_parameters,
                 partitioning=False,
                 hierarchy_required=False,
                 hierarchy=None,
                 run_id=1):
        super(RandomForestClassMethod, self).__init__(classifier=classifier,
                                                      classifier_params=classifier_params,
                                                      partitioning=partitioning, hierarchy_required=hierarchy_required,
                                                      hierarchy=hierarchy, run_id=run_id)

    @staticmethod
    def name():
        return "RF"

    def fit(self, X_train, y_train):
        self.partitions_for_node = (X_train, y_train)
        print("Running KNN Imputation")
        knn_start = time.time()
        imp = KNNImputer(missing_values=np.nan)
        X_train = imp.fit_transform(X_train)
        self.imp = imp
        knn_time = time.time() - knn_start
        print(f"Finished KNN Imputation, took {knn_time}s")
        self.stats_tracker.add_knn_time(knn_time)
        return self._fit_clf(X_train, y_train)

    def _fit_clf(self, X_train, y_train):
        model_start = time.time()
        fitted_clf = self.classifier.fit(X_train, y_train)
        model_time = time.time() - model_start
        self.stats_tracker.add_model_training_time(model_time)
        return fitted_clf

    def predict(self, df_test, **kwargs):
        df_test_numeric = df_test.select_dtypes(include=np.float)
        X_test = df_test_numeric[[f"F{i}" for i in range(df_test_numeric.shape[1])]].to_numpy()
        # X_test = df_test[[f"F{i}" for i in range(100)]]
        X_test = self.imp.transform(X_test)
        return self.classifier.predict(X_test)

    def predict_test_samples(self, df_test, e=10):
        df_test_numeric = df_test.select_dtypes(include=np.float)
        X_test = df_test_numeric[[f"F{i}" for i in range(df_test_numeric.shape[1])]].to_numpy()
        X_test = self.imp.transform(X_test)

        pred_start = time.time()
        y_pred = self.classifier.predict_proba(X_test)
        prediction_time = time.time() - pred_start
        self.stats_tracker.set_prediction_time(prediction_time)
        y_true = df_test['target'].to_numpy()

        self.stats_tracker.add_predictions(y_pred, y_true, e, df_test["group"], self.classifier.classes_)
        return {e: top_k_accuracy_score(y_true=y_true, y_score=y_pred, k=e, labels=self.classifier.classes_) for e in
                range(1, e + 1)}


class RandomForestBorutaMethod(RandomForestClassMethod):
    def __init__(self, classifier=RandomForestClassifier,
                 classifier_params=random_forest_parameters,
                 partitioning=False, hierarchy_required=False, hierarchy=None, run_id=1):
        super(RandomForestClassMethod, self).__init__(classifier=classifier,
                                                      classifier_params=classifier_params,
                                                      partitioning=partitioning, hierarchy_required=hierarchy_required,
                                                      hierarchy=hierarchy, run_id=run_id)

    def fit(self, X_train, y_train):
        self.partitions_for_node = (X_train, y_train)
        imp = KNNImputer(missing_values=np.nan)
        X_train = imp.fit_transform(X_train)
        self.imp = imp

        boruta_start = time.time()
        feat_selector = BorutaPy(self.classifier, n_estimators='auto', max_iter=100, verbose=2, random_state=1)
        feat_selector.fit(X_train, y_train)
        self.feat_selector = feat_selector
        X_train = feat_selector.transform(X_train, weak=True)
        boruta_time = time.time() - boruta_start
        print(f"selected feature shape: {X_train.shape}")
        self.stats_tracker.add_boruta_training_time(boruta_time)
        print(f"Training Time for Boruta is: {boruta_time}")

        return super().fit(X_train, y_train)

    def predict_test_samples(self, df_test, e=10):
        df_test_numeric = df_test.select_dtypes(include=np.float)
        X_test = df_test_numeric[[f"F{i}" for i in range(df_test_numeric.shape[1])]].to_numpy()
        y_true = df_test['target'].to_numpy()

        X_test = self.feat_selector.transform(X_test, weak=True)
        # Use already fitted KNN and Boruta --> Does not work if fit is not called previously!
        X_test = self.imp.transform(X_test)
        # X_test = X_test[:, feat_selector.support_]
        y_pred = self.classifier.predict_proba(X_test)

        # update predictions for each sample
        self.stats_tracker.add_predictions(y_pred, y_true, e, df_test["group"], self.classifier.classes_)

        accuracy_per_e = {}
        for i in range(1, e + 1):
            acc_i = top_k_accuracy_score(y_true, y_pred, k=i, labels=self.classifier.classes_)
            accuracy_per_e[i] = acc_i
        return accuracy_per_e

    @staticmethod
    def name():
        return "RF+B"


class SPH(ClassificationMethod):

    def __init__(self, hierarchy=HardCodedHierarchy().create_hardcoded_hierarchy(), min_samples_per_class=1,
                 max_info_loss=0.25, partitioning=True, run_id=1):

        super(SPH, self).__init__(hierarchy=hierarchy, partitioning=partitioning, run_id=run_id)
        self.max_info_loss = max_info_loss
        self.min_samples_per_class = min_samples_per_class
        self.parameters = {"max info loss": self.max_info_loss}
        self.surrogate_sets = []

    def _run_sph(self):
        sph_start = time.time()
        root_node = self.hierarchy
        partitions_for_node = {}

        # get leave nodes
        product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]
        sph_executed = 0

        while len(product_group_nodes) > 0:
            group_node = product_group_nodes.pop(0)
            node_id = group_node.node_id
            check_passed = False

            # node to traverse, if checks passed we go up in the hierarchy
            traverse_node = group_node
            while not check_passed:
                # check that SPH
                group_labels = traverse_node.training_labels
                group_data = traverse_node.training_data
                check_passed, sph_data, sph_labels, info_loss = self._SPH_checks(group_data, group_labels,
                                                                                 self.min_samples_per_class,
                                                                                 self.max_info_loss)
                if not check_passed:
                    sph_executed += 1
                    self.stats_tracker.add_surrogate_set(traverse_node.node_id)

                if not check_passed and traverse_node.parent:
                    print(
                        f"Using surrogate for {traverse_node.node_id}, which is {traverse_node.parent.node_id} "
                        f"with info loss {info_loss} and {len(np.unique(sph_labels))} class(es)")
                    traverse_node = traverse_node.parent
                else:
                    # if  np.unique(group_sample_labels):
                    group_node.sph_data = sph_data
                    group_node.sph_labels = sph_labels
                    partitions_for_node[node_id] = {"data": sph_data, "labels": sph_labels}

        self.stats_tracker.sph_count += sph_executed
        sph_time = time.time() - sph_start
        self.stats_tracker.set_sph_time(sph_time)

        return partitions_for_node, sph_executed

    def _SPH_checks(self, group_data, group_labels, min_samples_per_class=1, max_info_loss=0.25):
        ##############################################################################
        ######## First step: Remove samples, where the class occurs only once ########
        # get samples, lables for this product group
        group_samples = np.array(group_data)
        group_labels = np.array(group_labels)
        original_n_samples = len(group_samples)

        # get for each class how often it occurs in this product group
        class_count = Counter(group_labels)
        indices = []
        for index, label in enumerate(group_labels):
            if class_count[label] > min_samples_per_class:
                indices.append(index)
        min_c_group_labels = np.take(group_labels, indices, axis=0)
        min_c_group_samples = np.take(group_samples, indices, axis=0)

        assert len(min_c_group_samples) == len(indices)
        assert len(min_c_group_labels) == len(indices)

        ####### First step finished ###################################################
        ###############################################################################

        ###############################################################################
        ###### Second step: Check that two classes available ##########################
        # check that two classes are available after class removal
        two_class_avail = len(np.unique(min_c_group_labels)) > 1
        ###### Second step Finished ###################################################
        ###############################################################################

        ###############################################################################
        ###### Third step: Check that info loss less than 25% #########################
        info_loss = 1 - (len(min_c_group_samples) / original_n_samples)
        less_info_loss = info_loss < max_info_loss
        ####### Third step Finished ###################################################
        ##############################################################################

        check_passed = two_class_avail and less_info_loss
        return check_passed, min_c_group_samples, min_c_group_labels, info_loss

    def fit(self, X_train, y_train):
        """
        Fits the SPH model. This means that a model repository will be created, where for each leave node in the
        hierarchy, a Classifier will be trained. To this end, SPH also selects the suitable set of training data
        by also discovering parent nodes if too less samples for classes are available.

        :param X_train:
        :param y_train:
        :return: model_repository: Dictionary with key of group/node name and the value is a trained classifier model.
        """
        partitions_for_node, sph_executed = self._run_sph()
        self.partitions_for_node = partitions_for_node
        self.model_repository = self._build_model_repository(partitions_for_node)
        return self.model_repository

    def predict_test_samples(self, df_test, e=10):
        """
        :param df_test: Dataframe of test data. Is expected to have the features as columns "F0", ..., "F99" and the class
        values as "target" column. We also assume a "group" column that identifies the group for each sample.
        :param e: lengths of the recommendation list. Will use range(1, e+1), i.e., 1, ...,e.
        :return: accuracy_per_e: Dictionary that has range(1, e+1) as keys and the according accuracy scores as value
        """
        if self.model_repository:
            average_accuracy = 0
            accuracy_per_e = {k: 0 for k in range(1, e + 1)}

            # predict samples for each group
            # --> easier for prediction since the classifiers per group have different classes
            for group in df_test['group'].unique():
                test_group_df = df_test[df_test['group'] == group]
                df_test_numeric = test_group_df.select_dtypes(include=np.float)
                sample_data = df_test_numeric[[f"F{i}" for i in range(df_test_numeric.shape[1])]].to_numpy()
                # sample_data = test_group_df[[f"F{i}" for i in range(100)]].to_numpy()
                y_pred = self._predict_test_data_for_group(sample_data, group)
                labels = self.model_repository[group].classes_
                y_test_group = test_group_df["target"].to_numpy()

                # update predictions for this group, i.e., the prediction for each sample
                self.stats_tracker.add_predictions(y_pred, y_test_group, e, test_group_df["group"], labels)

                for y_true, y_score in zip(y_test_group, y_pred):

                    for k in range(1, e + 1):
                        # calculate the top_k accuracy for this sample
                        acc_per_e = self._get_top_k_accuracy_for_sample(y_true, y_score, labels, k)
                        accuracy_per_e[k] += acc_per_e / len(df_test)

            return accuracy_per_e
        else:
            logging.error("No Model repository built! Make sure to call fit!")

    @staticmethod
    def name():
        return "SPH"

    @staticmethod
    def _get_top_k_accuracy_for_sample(y_true, y_score, labels, k=1):
        if y_true in labels:
            if len(y_score) > 2:
                y_score = y_score.reshape(1, -1)
            else:
                # if we have two classes we need only the one with the higher score for the top_k_accuracy
                y_score = [y_score[0]]

            accuracy = top_k_accuracy_score([y_true], y_score=y_score, k=k,
                                            labels=labels)
        else:
            accuracy = 0
        return accuracy

    def _build_model_repository(self, partitions_for_node):
        model_repository = {}

        for node_id, value_dict in partitions_for_node.items():
            node_data = value_dict["data"]
            node_labels = value_dict["labels"]
            estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                  ("forest", RandomForestClassifier(**random_forest_parameters))])
            print(f"Training RF on node {node_id}")

            model_start = time.time()
            estimator.fit(node_data, node_labels)
            model_training_time = time.time() - model_start
            self.stats_tracker.add_model_training_time(model_training_time)

            model_repository[node_id] = estimator

        return model_repository

    def _predict_test_data_for_group(self, sample_data, group):
        return self.model_repository[group].predict_proba(sample_data)


class SPHandCPI(SPH):
    def __init__(self, hierarchy=HardCodedHierarchy().create_hardcoded_hierarchy(), min_samples_per_class=1,
                 max_info_loss=0.25, p_threshold=0.8, gini_threshold=0.3, partitioning=True, run_id=1):

        self.sph = super(SPHandCPI, self).__init__(hierarchy=hierarchy, partitioning=partitioning,
                                                   max_info_loss=max_info_loss, run_id=run_id)
        self.max_info_loss = max_info_loss
        self.min_samples_per_class = min_samples_per_class

        self.gini_threshold = gini_threshold
        self.p_threshold = p_threshold

        self.parameters["gini"] = self.gini_threshold
        self.parameters["p value"] = self.p_threshold

    def fit(self, X_train, y_train):
        """
        Fits the SPH model. This means that a model repository will be created, where for each leave node in the
        hierarchy, a Classifier will be trained. To this end, SPH also selects the suitable set of training data
        by also discovering parent nodes if too less samples for classes are available.

        :param X_train:
        :param y_train:
        :param min_samples_per_class:
        :param max_info_loss:
        :return: model_repository: Dictionary with key of group/node name and the value is a trained classifier model.
        """
        ## First apply SPH
        partitions_for_node, sph_executed = self._run_sph()
        self._build_model_repository(partitions_for_node)

        return self.model_repository

    def _build_model_repository(self, partitions_for_node):
        model_repository = {}
        for node_id, partition in partitions_for_node.items():
            partition_data = partition["data"]
            partition_labels = partition["labels"]
            cpi_data, cpi_labels = self._run_cpi(partition_data, partition_labels)

            if isinstance(cpi_data, tuple) and isinstance(cpi_labels, tuple):
                # minority and majority sets
                minority_partition = cpi_data[0]
                majority_partition = cpi_data[1]

                minority_labels = cpi_labels[0]
                majority_labels = cpi_labels[1]
                # train majority and minority learners
                minority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                               ("forest", RandomForestClassifier(**random_forest_parameters))])

                minority_estimator.fit(minority_partition, minority_labels)

                majority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                               ("forest", RandomForestClassifier(**random_forest_parameters))])

                model_start = time.time()
                majority_estimator.fit(majority_partition, majority_labels)
                model_time = time.time() - model_start
                self.stats_tracker.add_model_training_time(model_time)

                model_repository[node_id] = (minority_estimator, majority_estimator)

                partitions_for_node[node_id] = ({"data": minority_partition,
                                                 "labels": minority_labels},
                                                {"data": majority_partition, "labels": majority_labels})
            else:
                estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                      ("forest", RandomForestClassifier(**random_forest_parameters))])
                model_start = time.time()
                estimator.fit(cpi_data, cpi_labels)
                model_time = time.time() - model_start
                self.stats_tracker.add_model_training_time(model_time)

                model_repository[node_id] = estimator

        self.partitions_for_node = partitions_for_node
        self.model_repository = model_repository

    def _run_cpi(self, partition_data, partition_labels):
        cpi_start = time.time()
        ############# Detector: Check if gini threshold reached ###################################################
        # We now call gini function to calculate gini index
        gini_value = gini(partition_labels)

        if gini_value > self.gini_threshold:
            # increase cpi count in stats tracker
            self.stats_tracker.cpi_count += 1

            ######### If yes then divsior is executed, i.e., partition to minority and majority sampels ###########
            class_counter = Counter(partition_labels)
            print("partition min/majority")
            print(class_counter)
            class_freq = np.array(list(class_counter.values()))
            # calculate q quantile of classes
            class_threshold = np.quantile(class_freq, self.p_threshold)

            if len([label for label in partition_labels if class_counter[label] > class_threshold]) == 0:
                print(f"class threshold before: {class_threshold}")
                # class threhsold equals the maximum counter --> reduce it
                class_threshold = class_threshold - 1
                print(f"class threshold after: {class_threshold}")

            # divide according to q quantile value the partition into minority and majority samples
            minority_indices = [ind for ind, label in enumerate(partition_labels) if
                                class_counter[label] <= class_threshold]
            # should be the same as all data without minority classes
            majority_indices = [ind for ind, _ in enumerate(partition_labels) if ind not in minority_indices]
            print("class threshold: {}".format(class_threshold))

            # assert that we still have the same amount of samples and labels
            assert len(minority_indices) + len(majority_indices) == len(partition_labels)

            minority_data = np.take(partition_data, minority_indices, axis=0)
            majority_data = np.take(partition_data, majority_indices, axis=0)

            minority_labels = np.take(partition_labels, minority_indices, axis=0)
            majority_labels = np.take(partition_labels, majority_indices, axis=0)

            majority_label_count = len(np.unique(majority_labels))

            # check if majority set contains only one class
            if majority_label_count == 1:
                # Preprocessing: OvA if only one class present in a partition
                majority_data = np.concatenate([majority_data, minority_data])
                # add minority set to majority set again and relabel minority to "-1"
                new_minority_labels = np.array([-1 for x in minority_labels])
                majority_labels = np.concatenate([majority_labels, new_minority_labels])

            # We use tuples with minority on first and majority on second position
            cpi_data = (minority_data, majority_data)
            cpi_labels = (minority_labels, majority_labels)
            cpi_time = time.time() - cpi_start
            self.stats_tracker.add_cpi_time(cpi_time)
            return cpi_data, cpi_labels

        else:
            cpi_time = time.time() - cpi_start
            self.stats_tracker.add_cpi_time(cpi_time)
            return partition_data, partition_labels

    def predict_test_samples(self, df_test, e=10):
        """

        :param df_test: Dataframe of test data. Is expected to have the features as columns F0, ..., F99 and the class values as target column.
        :param e: lengths of the recommendation list. Will use range(1, e+1), i.e., 1, ...,e.
        :return: accuracy_per_e: Dictionary that has range(1, e+1) as keys and the according accuracy scores as value
        """
        if self.model_repository:
            accuracy_per_e = {k: 0 for k in range(1, e + 1)}

            # predict samples for each group
            # --> easier for prediction since the classifiers per group have different classes
            for group in df_test['group'].unique():

                test_group_df = df_test[df_test['group'] == group]
                df_test_numeric = test_group_df.select_dtypes(include=np.float)
                sample_data = df_test_numeric[[f"F{i}" for i in range(df_test_numeric.shape[1])]].to_numpy()
                clf = self.model_repository[group]

                for sample, y_true in zip(sample_data, test_group_df['target'].to_numpy()):
                    # For CPI we could have tuples of the form (minority_clf, majority_clf)
                    if isinstance(clf, tuple):
                        minority_clf = clf[0]
                        majority_clf = clf[1]

                        min_probas = minority_clf.predict_proba(sample.reshape(1, -1))[0]
                        majority_probas = majority_clf.predict_proba(sample.reshape(1, -1))[0]

                        majority_probas = [x for x in majority_probas if x != -1]
                        majority_classes = majority_clf["forest"].classes_
                        minority_classes = minority_clf["forest"].classes_

                        # filter out probs and class for -1
                        maj_prob_classes = list(zip(majority_probas, majority_classes))
                        maj_prob_classes = [x for x in maj_prob_classes if x[1] != -1]

                        majority_probas = [x[0] for x in maj_prob_classes]
                        majority_classes = [x[1] for x in maj_prob_classes]
                        merged_classes = [*minority_classes, *majority_classes]
                        assert len(merged_classes) == len(minority_classes) + len(majority_classes)
                        merged_probabilities = [*min_probas, *majority_probas]
                        assert len(merged_probabilities) == len(min_probas) + len(majority_probas)

                        # sort classes and probabilities by the probability
                        sorted_classes = [x for _, x in
                                          sorted(zip(merged_probabilities, merged_classes), key=lambda pair: pair[0])]
                        sorted_probas = [y for y, _ in
                                         sorted(zip(merged_probabilities, merged_classes), key=lambda pair: pair[0])]

                        # iterate through the confidence value list together with the class
                        for index in range(len(sorted_classes) - 1):

                            prob = sorted_probas[index]
                            cls = sorted_classes[index]

                            next_prob = sorted_probas[index + 1]
                            next_cls = sorted_classes[index + 1]

                            # difference in confidence value less than 1.5%?
                            if next_prob - prob < 0.015:
                                # minority class higher confidence than majority class? (lists are sorted)
                                if cls in majority_classes and next_cls in minority_classes:
                                    # only take into account if majority and minority confidence are higher than uniform random
                                    # probability
                                    if prob > 1 / len(majority_classes) and next_prob > 1 / len(minority_classes):
                                        # change confidence values
                                        prob = prob * 1.015
                                        next_prob = next_prob - next_prob * 0.015

                                        sorted_probas[index] = next_prob
                                        sorted_probas[index + 1] = prob
                                        sorted_classes[index] = next_cls
                                        sorted_classes[index + 1] = cls

                        sorted_zip_list = sorted(zip(sorted_probas, sorted_classes), key=lambda pair: pair[1])

                        labels = np.array([cls for _, cls in sorted_zip_list])
                        y_score = [[prob for prob, _ in sorted_zip_list]]
                    else:
                        labels = self.model_repository[group].classes_
                        y_score = self._predict_test_data_for_group(sample.reshape(1, -1), group)

                    # update predictions for this group, i.e., the prediction for each sample
                    self.stats_tracker.add_predictions(y_score, [y_true], e, [group], labels)

                    # only take first score as we have only one sample, but convert it to numpy arrray
                    y_score = np.array(y_score[0])
                    for k in range(1, e + 1):
                        accuracy_per_e[k] += self._get_top_k_accuracy_for_sample([y_true], y_score, k=k,
                                                                                 labels=labels) / len(df_test)

            return accuracy_per_e

        else:
            logging.error("No Model repository built! Make sure to call fit!")

    @staticmethod
    def name():
        return "SPH+CPI"


class CPI(SPHandCPI):
    def __init__(self, hierarchy=HardCodedHierarchy().create_hardcoded_hierarchy(), p_threshold=0.8,
                 gini_threshold=0.3, run_id=1):
        super(CPI, self).__init__(p_threshold=p_threshold, gini_threshold=gini_threshold, hierarchy=hierarchy,
                                  partitioning=True, run_id=run_id)
        del self.parameters["max info loss"]

    def fit(self, X_train, y_train):
        # cpi_data, cpi_labels = self._run_cpi(X_train, y_train)
        # Bad hack, we put for each leave node the whole data and train Random Forest. Not the best but shortest way.
        groups = [node for node in PreOrderIter(self.hierarchy) if not node.children]
        partitions_for_node = {group.node_id: {"data": X_train, "labels": y_train} for group in groups}

        self._build_model_repository(partitions_for_node)
        return self.model_repository

    @staticmethod
    def name():
        return "CPI"
