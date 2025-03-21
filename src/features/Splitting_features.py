from visualization.visualize import housing_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# split the data to representative to various categories of median incomes
# ------------------------------------------

# segmentation and sort data into discrete bins.
housing_df["median_categ"] = pd.cut(
    housing_df["median_income"],
    bins=[0, 1.5, 3, 4.5, 6, np.inf],
    labels=[1, 2, 3, 4, 5],
)

housing_df["median_categ"].hist(grid=False)
plt.title("Segmentation and sort data into discrete bins")
plt.savefig(
    "../../reports/figures/02_Histogram_of_income_median_categories.png",
    format="png",
    dpi=100,
)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df["median_categ"]):
    stratif_train_set = housing_df.loc[train_index]
    stratif_test_set = housing_df.loc[test_index]

"""
StratifiedShuffleSplit is used to split the dataset into training and testing sets, ensuring that the 
distribution of the 'median_categ' column (the categorical feature) is preserved in both sets.
The data is split into 80% training and 20% testing, while maintaining the same proportions of 
each category in both sets. The random_state ensures the split is reproducible.
"""

# compare income median category proportions in the test set and full data set
stratif_test_set["median_categ"].value_counts() / len(stratif_test_set)
housing_df["median_categ"].value_counts() / len(housing_df)

# drop income median category proportions
for set_ in (stratif_test_set, stratif_train_set):
    set_.drop("median_categ", axis=1, inplace=True)


def save_pickle_sets(
    stratif_train_set, stratif_test_set, sets_path="../../data/interim"
):

    os.makedirs(sets_path, exist_ok=True)

    train_pickle_path = os.path.join(sets_path, "stratif_train_set.pkl")
    stratif_train_set.to_pickle(train_pickle_path)

    test_pickle_path = os.path.join(sets_path, "stratif_test_set.pkl")
    stratif_test_set.to_pickle(test_pickle_path)

    print(f"Training and test sets saved to {sets_path}")


save_pickle_sets(stratif_train_set, stratif_test_set)
