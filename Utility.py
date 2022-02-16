from collections import Counter

import numpy as np
from anytree import PreOrderIter
from sklearn.model_selection import train_test_split


def get_train_test_X_y(df_train, df_test, n_features=100):
    X_train = df_train[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_test = df_test["target"].to_numpy()
    return X_train, X_test, y_train, y_test


def train_test_splitting(df, n_train_samples=750, at_least_two_samples=True):
    """
    Performs our custom train test splitting.
    This highly affects the accuracy of this approach.

    :param df: Dataframe of the whole data.
    :param n_train_samples: number of samples to use for training
    :param at_least_two_samples: check if we should have at least one sample in the training data for each group.
    :return: train, test: dataframes that contain training and test data.
    """
    # split in 750 training samples
    # 300 test samples
    train_percent = n_train_samples / len(df)

    n_classes = len(np.unique(df["target"].to_numpy()))

    counter = Counter(df["target"].to_numpy())
    try:
        # split with stratify such that each class occurs in train and test set
        train, test = train_test_split(df, train_size=train_percent, random_state=1234,
                                       stratify=df["target"]
                                       )
    except ValueError:
        train, test = train_test_split(df, train_size=train_percent, random_state=1234,
                                       #stratify=df["target"]
                                       )
    n_not_in_train = 1

    while at_least_two_samples and n_not_in_train > 0:
        test['freq'] = test.groupby('group')['target'].transform('count')
        train['freq'] = train.groupby('group')['target'].transform('count')

        # check which classes occur once and do not occur on training data!
        # Use counter for this of group and target, in both training and test set
        train_counter = Counter(zip(train['target'].to_numpy(), train['group'].values))

        # mark which ones occur not in training set (group and target) but occur in test set
        test['marker'] = test.apply(lambda row: train_counter[(row['target'], row['group'])] == 0, axis=1)
        test_in_train = test[~test['marker']]
        test_not_in_train = test[test['marker']]

        n_not_in_train = len(test_not_in_train)

        # check if there is still a class that occurs in test but not in training
        if n_not_in_train > 0:
            # if yes, replace these samples with random samples from training set
            drop_indices = np.random.choice(train.index, n_not_in_train,
                                            # p=probability
                                            )
            train_subset = train.loc[drop_indices]
            train = train.drop(drop_indices)

            train = train.append(test_not_in_train)
            test = test_in_train.append(train_subset)

        test = test.drop(['freq'], axis=1)
        test = test.drop(['marker'], axis=1)
        train = train.drop(['freq'], axis=1)
        if "marker" in train.columns:
            train = train.drop(['marker'], axis=1)

    return train, test


def update_data_and_training_data(root_node, df_train, n_features=100):
    product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]

    for group_node in product_group_nodes:
        # we have leave node so set training data
        node_id = group_node.node_id

        # get training data and training labels as numpy arrays
        train_data = df_train[df_train["group"] == node_id][[f"F{i}" for i in range(n_features)]].to_numpy()
        training_labels = df_train[df_train["group"] == node_id]["target"].to_numpy()


        group_node.training_data = train_data
        group_node.training_labels = training_labels

        # also set the "medium" data and labels
        # These are more than the training data and in case we already created a dataset, we want to set them as well

        # pass training data upwards the whole tree
        traverse_node = group_node
        while traverse_node.parent:
            traverse_node = traverse_node.parent

            if traverse_node.training_data is not None:
                traverse_node.training_data = np.concatenate([traverse_node.training_data, train_data])
                traverse_node.training_labels = np.concatenate([traverse_node.training_labels, training_labels])
            else:
                traverse_node.training_data = train_data
                traverse_node.training_labels = training_labels
    return root_node
