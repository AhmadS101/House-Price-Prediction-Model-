import os
import pandas as pd
import numpy as np

# -----------------------------------------------------------
# Prepare the transformation pipeline
# -----------------------------------------------------------


# load training data
def load_train_housing_data(housing_path="../../data/interim"):
    pkl_path = os.path.join(housing_path, "stratif_train_set.pkl")
    return pd.read_pickle(pkl_path)


train_set = load_train_housing_data()

# specifi the predictors and the target feature
housing = train_set.drop("median_house_value", axis=1)
housing_target = train_set["median_house_value"].copy()


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

housing_numeric = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_numeric)
categ_attribs = ["ocean_proximity"]

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)


full_pipeline = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, num_attribs),
        ("categorical", OneHotEncoder(), categ_attribs),
    ]
)


housing_preproc = full_pipeline.fit_transform(housing)

# -----------------------------------------------------------
# Training our data with virous models
# -----------------------------------------------------------

#  tray with distance base model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_preproc, housing_target)

# test the model prediction against targeted feature values
some_data = housing.iloc[:5]
some_labels = housing_target.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print(f"prediction prices: {lin_reg.predict(some_data_prepared)}")
print(f"targeted: {list(some_labels)}")

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_preproc)
lin_mse = mean_squared_error(housing_target, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
""" 
prediction error of $68,628 is not very satisfying. 
The model underfitting the training data.mean that the features do not provide enough information 
to make good predictions, or that the model is not powerful enough.
"""
# tray with another tree model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_preproc, housing_target)

housing_tree_predictions = tree_reg.predict(housing_preproc)
tree_mse = mean_squared_error(housing_target, housing_tree_predictions)
tree_rmse = np.sqrt(tree_mse)
""" 
No error at all? Could this model really be absolutely perfect? 
it is much more likely that the model has badly overfit the data.
due to this prediction we need to test our model with a valation sets
"""

# -----------------------------------------------------------
# Training our data with virous models
# -----------------------------------------------------------

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    tree_reg, housing_preproc, housing_target, scoring="neg_mean_squared_error", cv=10
)
tree_rmse_scores = np.sqrt(-scores)


# function for printing the root_mean_squared_error
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)

""" 
The Decision Tree has ascore of approximately 71,609, generally ±2,449. 
Now the Decision Tree doesn’t look as good as it did earlier. it seems to perform worse than the Linear Regression model
"""

# Cross-Validation for LR
lin_scores = cross_val_score(
    lin_reg, housing_preproc, housing_target, scoring="neg_mean_squared_error", cv=10
)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
""" 
the linear regression model changed with slightly low values so it's still under fitted  
"""
# tray with another ensemble model and it's cross-validation
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_preproc, housing_target)
forest_pred = forest_reg.predict(housing_preproc)
forest_mse = mean_squared_error(housing_target, forest_pred)
forest_rmse = np.sqrt(forest_mse)

forest_scores = cross_val_score(
    forest_reg, housing_preproc, housing_target, scoring="neg_mean_squared_error", cv=10
)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

""" 
Random Forests look very promising with mean of  18,687$.
event that the score on the training set is still much lower than on the validation sets $50,182, 
meaning that the model is still overfitting the training set
"""

# -----------------------------------------------------------
# Fine-Tunning RandomForest model
# -----------------------------------------------------------

params = [
    {"n_estimators": [3, 10, 30, 35, 40], "max_features": [2, 4, 6, 8, 10]},
    {"bootstrap": [False], "n_estimators": [3, 10, 30], "max_features": [2, 3, 4]},
]

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    forest_reg, params, cv=5, scoring="neg_mean_squared_error", return_train_score=True
)
grid_search.fit(housing_preproc, housing_target)

grid_search.best_params_  # 'max_features': 4, 'n_estimators': 30 we can make the model better by increasing the n_estimators
grid_search.best_estimator_

#  evaluation scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(f"Mean Score: {np.sqrt(-mean_score)} Best Params: {params}")

""" 
the max_features hyperparameter to 4 and the n_estimators hyperparameter to 30. 
The RMSE score for this combination is $49,241, which is slightly better than the score you got earlier 
default hyperparameter values (which was $50,352)using the
"""


# -----------------------------------------------------------
# Evaluate Our System on the Test Set
# -----------------------------------------------------------


def load_test_housing_data(housing_path="../../data/interim"):
    pkl_path = os.path.join(housing_path, "stratif_test_set.pkl")
    return pd.read_pickle(pkl_path)


test_set = load_test_housing_data()

final_model = grid_search.best_estimator_

x_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()


x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)  # evaluates to 46,987


#  compute a 95% confidence interval
from scipy import stats

confidence = 0.95
squared_error = (final_predictions - y_test) ** 2
np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_error) - 1,
        loc=squared_error.mean(),
        scale=stats.sem(squared_error),
    )
)
""" 
the 95% confidence interval of our model prediction is [$45,044, $48,853]
"""

# -----------------------------------------------------------
# saving the models
# -----------------------------------------------------------

import joblib


def save_model(model, model_path="../../models"):
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, "final_model.pkl")
    joblib.dump(model, model_path)


save_model(final_model)
