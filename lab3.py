# ============================================================
# LAB SHEET-03
# SUPERVISED LEARNING - REGRESSION MODELS
# MCA III SEMESTER
# ============================================================

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.datasets import fetch_california_housing


# ============================================================
# PROGRAM 1: Load a Regression Dataset
# ============================================================

data = pd.read_csv("housing.csv")

print(data.head())


# ============================================================
# PROGRAM 2: Display First and Last Five Records
# ============================================================

print("First Five Records:")
print(data.head())

print("Last Five Records:")
print(data.tail())


# ============================================================
# PROGRAM 3: Dataset Information and Descriptive Statistics
# ============================================================

print("Dataset Information:")
data.info()

print("Descriptive Statistics:")
print(data.describe())


# ============================================================
# PROGRAM 4: Identify Input and Output Variables
# ============================================================

X = data.drop("price", axis=1)
y = data["price"]

print("Independent Variables:")
print(X.columns)

print("Dependent Variable:")
print(y.name)


# ============================================================
# PROGRAM 5: Split Dataset into Training and Testing Sets
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ============================================================
# PROGRAM 6: Implement Simple Linear Regression
# ============================================================

X_simple = data[["area"]]
y_simple = data["price"]

simple_model = LinearRegression()

print("Simple Linear Regression Model Created")


# ============================================================
# PROGRAM 7: Train Linear Regression Model
# ============================================================

X_train_simple, X_test_simple, y_train_simple, y_test_simple = (
    train_test_split(
        X_simple,
        y_simple,
        test_size=0.2,
        random_state=42
    )
)

simple_model.fit(X_train_simple, y_train_simple)

print("Linear Regression Model Trained")


# ============================================================
# PROGRAM 8: Predict Output Values
# ============================================================

simple_predictions = simple_model.predict(X_test_simple)

print("Predicted Values:")
print(simple_predictions)


# ============================================================
# PROGRAM 9: Visualize Linear Regression Line
# ============================================================

simple_model_all = LinearRegression()
simple_model_all.fit(X_simple, y_simple)

plt.scatter(X_simple, y_simple)
plt.plot(
    X_simple,
    simple_model_all.predict(X_simple)
)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.show()


# ============================================================
# PROGRAM 10: Compare Actual and Predicted Values
# ============================================================

comparison = pd.DataFrame({
    "Actual": y_test_simple,
    "Predicted": simple_predictions
})

print(comparison)


# ============================================================
# PROGRAM 11: Display Regression Coefficient and Intercept
# ============================================================

print("Coefficient:", simple_model.coef_)
print("Intercept:", simple_model.intercept_)


# ============================================================
# PROGRAM 12: Predict Output for New Input
# ============================================================

new_area = [[2000]]

new_prediction = simple_model.predict(new_area)

print("Predicted Price:", new_prediction)


# ============================================================
# PROGRAM 13: Multiple Linear Regression
# ============================================================

X_multiple = data[
    ["area", "bedrooms", "bathrooms"]
]

y_multiple = data["price"]

multiple_model = LinearRegression()
multiple_model.fit(X_multiple, y_multiple)

print("Multiple Linear Regression Implemented")


# ============================================================
# PROGRAM 14: Train Multiple Linear Regression Model
# ============================================================

X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = (
    train_test_split(
        X_multiple,
        y_multiple,
        test_size=0.2,
        random_state=42
    )
)

multiple_model.fit(
    X_train_multiple,
    y_train_multiple
)

print("Multiple Regression Model Trained")


# ============================================================
# PROGRAM 15: Predict Using Testing Dataset
# ============================================================

multiple_predictions = multiple_model.predict(
    X_test_multiple
)

print("Predicted Values:")
print(multiple_predictions)


# ============================================================
# PROGRAM 16: Graphical Comparison
# ============================================================

plt.scatter(
    y_test_multiple,
    multiple_predictions
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


# ============================================================
# PROGRAM 17: Analyze Effect of Independent Variables
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X_multiple.columns,
    "Coefficient": multiple_model.coef_
})

print(coefficients)


# ============================================================
# PROGRAM 18: Polynomial Regression Degree 2
# ============================================================

poly_degree_2 = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_degree_2.fit(
    X_simple,
    y_simple
)

print("Polynomial Regression Degree 2 Implemented")


# ============================================================
# PROGRAM 19: Polynomial Regression Degree 3
# ============================================================

poly_degree_3 = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

poly_degree_3.fit(
    X_simple,
    y_simple
)

print("Polynomial Regression Degree 3 Implemented")


# ============================================================
# PROGRAM 20: Compare Linear and Polynomial Regression
# ============================================================

linear_model = LinearRegression()
linear_model.fit(X_simple, y_simple)

linear_prediction = linear_model.predict(X_simple)

poly_prediction = poly_degree_2.predict(X_simple)

linear_r2 = r2_score(
    y_simple,
    linear_prediction
)

poly_r2 = r2_score(
    y_simple,
    poly_prediction
)

print("Linear Regression R2:", linear_r2)
print("Polynomial Regression R2:", poly_r2)


# ============================================================
# PROGRAM 21: Visualize Polynomial Regression Curve
# ============================================================

poly_features = PolynomialFeatures(degree=2)

X_poly = poly_features.fit_transform(
    X_simple
)

poly_model = LinearRegression()
poly_model.fit(
    X_poly,
    y_simple
)

X_range = np.linspace(
    X_simple.min(),
    X_simple.max(),
    100
).reshape(-1, 1)

X_range_poly = poly_features.transform(
    X_range
)

y_range = poly_model.predict(
    X_range_poly
)

plt.scatter(
    X_simple,
    y_simple
)

plt.plot(
    X_range,
    y_range
)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Polynomial Regression Curve")
plt.show()


# ============================================================
# PROGRAM 22: Predict Using Polynomial Regression
# ============================================================

new_area_poly = [[2000]]

new_area_transformed = poly_features.transform(
    new_area_poly
)

poly_new_prediction = poly_model.predict(
    new_area_transformed
)

print("Polynomial Prediction:")
print(poly_new_prediction)


# ============================================================
# PROGRAM 23: Compare Different Polynomial Degrees
# ============================================================

for degree in [1, 2, 3, 4, 5]:

    polynomial = PolynomialFeatures(
        degree=degree
    )

    X_degree = polynomial.fit_transform(
        X_simple
    )

    model_degree = LinearRegression()

    model_degree.fit(
        X_degree,
        y_simple
    )

    prediction_degree = model_degree.predict(
        X_degree
    )

    score_degree = r2_score(
        y_simple,
        prediction_degree
    )

    print(
        "Degree:",
        degree,
        "R2 Score:",
        score_degree
    )


# ============================================================
# PROGRAM 24: Mean Absolute Error
# ============================================================

mae = mean_absolute_error(
    y_test_simple,
    simple_predictions
)

print("MAE:", mae)


# ============================================================
# PROGRAM 25: Mean Squared Error
# ============================================================

mse = mean_squared_error(
    y_test_simple,
    simple_predictions
)

print("MSE:", mse)


# ============================================================
# PROGRAM 26: Root Mean Squared Error
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test_simple,
        simple_predictions
    )
)

print("RMSE:", rmse)
# PROGRAM 27: R-Squared Score

r2 = r2_score(
    y_test_simple,
    simple_predictions
)

print("R2 Score:", r2)


# ============================================================
# PROGRAM 28: Compare Linear and Polynomial Metrics
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_simple,
    y_train_simple
)

linear_test_prediction = linear_model.predict(
    X_test_simple
)

poly_features_compare = PolynomialFeatures(
    degree=2
)

X_train_poly = poly_features_compare.fit_transform(
    X_train_simple
)

X_test_poly = poly_features_compare.transform(
    X_test_simple
)

poly_compare_model = LinearRegression()

poly_compare_model.fit(
    X_train_poly,
    y_train_simple
)

poly_test_prediction = poly_compare_model.predict(
    X_test_poly
)

print("Linear Regression Metrics:")

print(
    "MAE:",
    mean_absolute_error(
        y_test_simple,
        linear_test_prediction
    )
)

print(
    "MSE:",
    mean_squared_error(
        y_test_simple,
        linear_test_prediction
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test_simple,
            linear_test_prediction
        )
    )
)

print(
    "R2:",
    r2_score(
        y_test_simple,
        linear_test_prediction
    )
)

print("Polynomial Regression Metrics:")

print(
    "MAE:",
    mean_absolute_error(
        y_test_simple,
        poly_test_prediction
    )
)

print(
    "MSE:",
    mean_squared_error(
        y_test_simple,
        poly_test_prediction
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test_simple,
            poly_test_prediction
        )
    )
)

print(
    "R2:",
    r2_score(
        y_test_simple,
        poly_test_prediction
    )
)


# ============================================================
# PROGRAM 29: Interpret MSE and R2
# ============================================================

print("MSE represents the average squared prediction error.")

print(
    "Lower MSE indicates better prediction performance."
)

print(
    "R2 represents how much variation in the dependent "
    "variable is explained by the model."
)

if r2 > 0.7:
    print("The model has good explanatory power.")
else:
    print("The model needs improvement.")


# ============================================================
# PROGRAM 30: Visualize Prediction Errors
# ============================================================

prediction_errors = (
    y_test_simple - simple_predictions
)

plt.scatter(
    simple_predictions,
    prediction_errors
)

plt.axhline(0)

plt.xlabel("Predicted Values")
plt.ylabel("Prediction Errors")
plt.title("Prediction Error Plot")

plt.show()


# ============================================================
# PROGRAM 31: Regression Using Standardized Features
# ============================================================

X_standard = data[
    ["area", "bedrooms", "bathrooms"]
]

y_standard = data["price"]

X_train_standard, X_test_standard, y_train_standard, y_test_standard = (
    train_test_split(
        X_standard,
        y_standard,
        test_size=0.2,
        random_state=42
    )
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_standard
)

X_test_scaled = scaler.transform(
    X_test_standard
)

standard_model = LinearRegression()

standard_model.fit(
    X_train_scaled,
    y_train_standard
)

standard_predictions = standard_model.predict(
    X_test_scaled
)

print("Standardized Regression Completed")


# ============================================================
# PROGRAM 32: Compare Before and After Feature Scaling
# ============================================================

before_scaling_model = LinearRegression()

before_scaling_model.fit(
    X_train_standard,
    y_train_standard
)

before_prediction = before_scaling_model.predict(
    X_test_standard
)

before_r2 = r2_score(
    y_test_standard,
    before_prediction
)

after_scaling_model = LinearRegression()

after_scaling_model.fit(
    X_train_scaled,
    y_train_standard
)

after_prediction = after_scaling_model.predict(
    X_test_scaled
)

after_r2 = r2_score(
    y_test_standard,
    after_prediction
)

print("R2 Before Scaling:", before_r2)
print("R2 After Scaling:", after_r2)


# ============================================================
# PROGRAM 33: Regression Using Another Real-World Dataset
# ============================================================

california_housing = fetch_california_housing(
    as_frame=True
)

california_X = california_housing.data
california_y = california_housing.target

california_X_train, california_X_test, california_y_train, california_y_test = (
    train_test_split(
        california_X,
        california_y,
        test_size=0.2,
        random_state=42
    )
)

california_model = LinearRegression()

california_model.fit(
    california_X_train,
    california_y_train
)

california_prediction = california_model.predict(
    california_X_test
)

print(california_prediction)


# ============================================================
# PROGRAM 34: Save Trained Regression Model Using Joblib
# ============================================================

model_to_save = LinearRegression()

model_to_save.fit(
    X_simple,
    y_simple
)

joblib.dump(
    model_to_save,
    "linear_regression_model.pkl"
)

print("Model saved successfully.")


# ============================================================
# PROGRAM 35: Load Saved Model and Predict New Data
# ============================================================

loaded_model = joblib.load(
    "linear_regression_model.pkl"
)

new_input = [[2000]]

loaded_prediction = loaded_model.predict(
    new_input
)

print("Predicted Value:")
print(loaded_prediction)
