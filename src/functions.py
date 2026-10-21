import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


# For main.py, to calculate the optimal pH and temperature #
def calculate_average(values):
    return sum(values) / len(values)


# For classes.py, model training #
# loads and splits datas #
def data_loading():
    df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_4factor.csv"))

    features = df.columns[1:]
    targets = df.columns[0]

    x = df.loc[:, features].to_numpy()
    y = df.loc[:, targets].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test

# Polynomial model pipeline: includes preprocessing (standardization), and hyperparameter optimization (GridSearchCV) #
def train_polynomial_model(x_train, y_train):
    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )

    # Polynomial degrees 1,2,3,4 and 5 were included in this model training #
    param_grid = dict(polynomialfeatures__degree=[i for i in range(1, 6)])

    poly_model_gscv = GridSearchCV(poly_model,
                                   param_grid=param_grid,
                                   cv=5, scoring="neg_mean_squared_error",
                                   return_train_score=True
                                   )

    poly_model_gscv.fit(x_train, y_train)
    best_model = poly_model_gscv.best_estimator_
    best_params = poly_model_gscv.best_params_
    return poly_model_gscv, best_params, best_model

# KNN model training #
def train_knn_model(x_train, y_train):
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )

    # Using GridSearchCV, k values of 3,5,7,9,11 and 15 were searched to find the optimal k #
    param_grid = dict(
        kneighborsregressor__n_neighbors=[3, 5, 7, 9, 11, 15],
        kneighborsregressor__weights=["uniform", "distance"],
    )

    knn_model_gscv = GridSearchCV(knn_model,
                                  param_grid=param_grid,
                                  cv=5, scoring="neg_mean_squared_error",
                                  return_train_score=True
                                  )

    knn_model_gscv.fit(x_train, y_train)
    best_model = knn_model_gscv.best_estimator_
    best_params = knn_model_gscv.best_params_

    return knn_model_gscv, best_params, best_model

# Test fold of the workflow #
def evaluate_model(best_model, x_test, y_test):
    y_predict = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

    return rmse, r2, y_predict

# Exports values predicted by the model #
def prediction_results(y_test, y_predict, filename="predicted_rate.csv"):
    predict_df = pd.DataFrame({
        "Actual Growth Rate[d^-1]": y_test,
        "Predicted Growth Rate[d^-1]": y_predict
    })

    export_results = os.path.join("datasets", filename)
    predict_df.to_csv(export_results, index=False)

    return export_results


# Plot Formatting #
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12


def plot_validation_curve(param_name, xlabel, title, filename, grid_search_result):
    results = pd.DataFrame(grid_search_result.cv_results_)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

    # KNN regression verification curve #
    if "param_kneighborsregressor__weights" in results.columns:
        df_uniform = results[results["param_kneighborsregressor__weights"] == "uniform"]
        df_distance = results[results["param_kneighborsregressor__weights"] == "distance"]

        # Plot Uniform Curves #
        k_uniform = df_uniform.loc[:, f"param_{param_name}"].to_numpy()
        ax.plot(k_uniform,
                np.sqrt(-df_uniform.loc[:, "mean_train_score"].to_numpy()),
                linestyle="-",
                color=COLORS[0],
                label="Train (KNN: uniform)"
                )
        ax.plot(k_uniform,
                np.sqrt(-df_uniform.loc[:, "mean_test_score"].to_numpy()),
                linestyle="--",
                color=COLORS[0],
                label="Valid (KNN: uniform)"
                )

        # Plot Distance Curves #
        k_distance = df_distance.loc[:, f"param_{param_name}"].to_numpy()
        ax.plot(k_distance,
                np.sqrt(-df_distance.loc[:, "mean_train_score"].to_numpy()),
                linestyle="-",
                color=COLORS[1],
                label="Train (KNN: distance)"
                )
        ax.plot(k_distance,
                np.sqrt(-df_distance.loc[:, "mean_test_score"].to_numpy()),
                linestyle="--",
                color=COLORS[1],
                label="Valid (KNN: distance)"
                )

    # Polynomial regression curve #
    else:
        degrees = results.loc[:, f"param_{param_name}"].to_numpy()
        rmse_train = np.sqrt(-results.loc[:, "mean_train_score"].to_numpy())
        rmse_test = np.sqrt(-results.loc[:, "mean_test_score"].to_numpy())

        ax.plot(degrees, rmse_train, linestyle="-", color=COLORS[2], label="Train (Poly)")
        ax.plot(degrees, rmse_test, linestyle="--", color=COLORS[3], label="Valid (Poly)")

    # Plot Formatting #
    ax.set_xlabel(xlabel, fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(labelsize=fontsize_ticks, axis='both', which='major')
    ax.legend(fontsize=fontsize_legend)
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    fig_path = os.path.join("figures", filename)
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


