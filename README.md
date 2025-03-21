# House-Price-Prediction
### Project Overview

- This project focuses on applying machine learning regression techniques to predict house prices. The primary objective is to develop a model that estimates house prices using features such as location, size, number of bedrooms, and other relevant factors.
---

###  Frame the Problem
- The business objective is to create a model that predicts a district's median housing price, which will be used by a downstream machine learning system to assess investment opportunities. The current manual estimation method is inaccurate and time-consuming. The goal is to build a supervised learning model, specifically a regression model, using census data to improve prediction accuracy. This is a multiple regression task, utilizing various features like population and income, and will be tackled using batch learning, as the dataset is manageable and does not require rapid adjustments.

- The final performance of the system achieved an accuracy of 40.8%, which is not better than the experts’ manual price estimates (often off by about 20%). However, launching the model may still be beneficial, as it could free up time for the experts to focus on more interesting and productive tasks.
---

### Installation

- To get started with this project, follow these steps:
#### **1. Clone the repository:**
```python
git clone https://github.com/AhmadS101/House-Price-Prediction-Model-.git
```
#### **2. Install the required dependencies:**
```python
pip install -r requirements.txt
```
---

### Usage

- To use this project, follow these steps:
#### **1. Download Dataset:**
```pyhton
python src\data\download_dataset.py
```

#### **2. Data Visualization for Caught Insights:**
```pyhton
python src\visualization\visualize.py
```

#### **3. Splitting Dataset:**
```pyhton
python src\features\Splitting_features.py
```

#### **4. Train Various Regression Models:** 
```pyhton
python src\models\train_model.py
```
---

### Dataset
- The dataset used in this project is sourced from [GitHub](https://github.com/ageron/handson-ml2). It contains information about various houses, including features such as:
Location
- Number of bedrooms
- Number of rooms
- Number of households
- Price

The dataset is stored in the data\raw directory.
---

### Model
- We tried various types of models, ranging from simple ones to more robust models, such as the linear regression model. However, the linear regression model caused significant underfitting of the data. Next, we used a tree-based model, the Decision Tree, which showed slight overfitting and resulted in a 0 mean square error. To improve this, we performed cross-validation on our training dataset. Afterward, we applied the ensemble model, Random Forest, with cross-validation, and it appeared to be the most promising model. To further improve its performance, we used GridSearchCV for hyperparameter tuning.

- We identified the following hyperparameters:

    1. max_features: 4
    2. n_estimators: 30
    3. bootstrap: [False]
---

### Evaluation
- The model's performance is evaluated using the following metrics:

Root Mean Absolute Error (RMAE): $46,987
---

### Results
- The final performance of the system achieved an accuracy of 40.8%, which is not better than the experts’ manual price estimates (often off by about 20%). However, launching the model may still be beneficial, as it could free up time for the experts to focus on more interesting and productive tasks.
