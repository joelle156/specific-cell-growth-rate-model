import src.functions as sf

class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")

        self.run_ml_pipeline()

    def run_ml_pipeline(self):
        x_train, x_test, y_train, y_test = sf.data_loading()

        poly_gscv, best_poly_params, best_poly_model = sf.train_polynomial_model(x_train, y_train)
        print(f"\nBest Polynomial Parameters: {best_poly_params}")

        knn_gscv, best_knn_params, best_knn_model = sf.train_knn_model(x_train, y_train)
        print(f"Best KNN Parameters: {best_knn_params}")

        poly_rmse, poly_r2, poly_pred = sf.evaluate_model(best_poly_model, x_test, y_test)
        print(f"Polynomial Test RMSE: {poly_rmse:.4f} | R2: {poly_r2:.4f}")

        knn_rmse, knn_r2, knn_pred = sf.evaluate_model(best_knn_model, x_test, y_test)
        print(f"KNN Test RMSE: {knn_rmse:.4f} | R2: {knn_r2:.4f}")

        sf.prediction_results(y_test, poly_pred, filename = "poly_prediction.csv")
        sf.prediction_results(y_test, knn_pred, filename = "knn_prediction.csv")

        sf.plot_validation_curve(
            param_name = "polynomialfeatures__degree",
            xlabel = "Polynomial degree",
            title = "Polynomial Regression Validation Curve",
            filename = "validation_curve_polynomial.png",
            grid_search_result = poly_gscv
        )

        sf.plot_validation_curve(
            param_name = "kneighborsregressor__n_neighbors",
            xlabel = "Number of Neighbors",
            title = "KNN Regression Validation Curve",
            filename = "validation_curve_knn.png",
            grid_search_result = knn_gscv
        )
        print("\nExecution of the Pipeline is complete. The model has been trained and evaluated successfully. Figures and predictions have been saved.\n")
