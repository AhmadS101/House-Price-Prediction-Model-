import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_housing_data(housing_path="../../data/raw/housing"):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing_df = load_housing_data()

# Exploare the data
housing_df.info()
# check for missing values
housing_df.isnull().sum()  # total bedrooms has 207 missing values
# check for categorical and numerical variable
numerical_features = [
    col for col in housing_df.columns if np.issubdtype(housing_df[col].dtype, np.number)
]

categorical_features = [
    col for col in housing_df.columns if pd.api.types.is_string_dtype(housing_df[col])
]  # ocean_proximity is the only categorical feature

# explore the categorical and numerical features values
housing_df[categorical_features].value_counts()
housing_df.describe().T

plt.figure(figsize=(25, 40))
for i, col in enumerate(numerical_features):
    plt.subplot(5, 2, i + 1)
    # i+1 is the index of the subplot, cause the enumerate start from 0
    sns.distplot(housing_df[col])
plt.savefig(
    "../../reports/figures/01_numerical_features_distribution.png",
    format="png",
    dpi=100,
)
"""
1-Some numerical features have been scaled and capped (e.g., housing median age, median house value, median income).
The median house value is the target you're predicting, and capping it could limit the model's ability to predict higher prices. 
Check with the client if predicting prices above the cap is important. If so, either collect more data for higher-value areas 
or remove capped areas from the datasets.

2-Many histograms exhibit a heavy right tail, make it harder for some machine learning algorithms to find patterns in the data 
indicating the need to transform these attributes later to achieve more bell-shaped distributions.

3-The different attributes (like income and house value) are on very different scales.
"""

# ---------------------------------------------
# Continue with train data only
# ---------------------------------------------
"""
lets take a Test_Set from the whole dataset, It might seem odd to set aside part of the data at this early stage,
our brains are a powerful pattern recognition system, which can also make it prone to overfitting. 
If you examine the test set too early, you might find an interesting pattern that influences your choice of machine learning model.
"""
housing = pd.read_pickle("../../data/interim/stratif_train_set.pkl")

housing.plot(kind="scatter", x="longitude", y="latitude")
plt.title(" A geographical scatterplot of the data")
plt.savefig(
    "../../reports/figures/03_ A_geographical_scatterplot_of_the_data.png",
    format="png",
    dpi=100,
)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.title("High density area of the data")
plt.savefig(
    "../../reports/figures/04_Highlights_high_density_area_of_the_data.png",
    format="png",
    dpi=100,
)

housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.1,
    s=housing["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()
plt.title(
    "red is expensive, blue is cheap, larger circles indicate areas with a larger population"
)
plt.savefig(
    "../../reports/figures/05_Highlights_population_and_high_prices_area_of_the_data.png",
    format="png",
    dpi=100,
)
""" 
The previous image tells you that the housing prices are very much related to the location
(close to the ocean) and to the population density.
"""

#  Looking for Correlations

corr_matrix = housing[numerical_features].corr()
corr_matrix["median_house_value"].sort_values(
    ascending=False
)  # strong positive correlation between median income and  median house value
attributes_check = [
    "median_house_value",
    "housing_median_age",
    "total_rooms",
    "median_income",
]

sns.pairplot(housing[attributes_check])
plt.title("Highly correlated numeric features")
plt.savefig(
    "../../reports/figures/06_Highly_correlated_numeric_feature.png",
    format="png",
    dpi=100,
)

# trying out various attribute combinations to increase the corrolation between them
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

attributes_check = [
    "median_house_value",
    "population_per_household",
    "bedrooms_per_room",
    "rooms_per_household",
]
corr_matrix = housing[attributes_check].corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
