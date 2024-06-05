from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error , root_mean_squared_error, mean_absolute_error

forest_model = RandomForestRegressor(n_estimators = 100, random_state=42)
forest_model.fit(train_data, train_labels)

test_pred = forest_model.predict(test_data[:5])
forest_mse = mean_squared_error(actual_val, test_pred)
forest_rmse = root_mean_squared_error(actual_val, test_pred)
forest_mae = mean_absolute_error(actual_val, test_pred)
forest_rmse

forest_scores = cross_val_score(forest_model, test_data,test_labels, scoring = "neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores : ",forest_rmse_scores)
print("Mean : ", forest_rmse_scores.mean())
print("Standard deviation : ", forest_rmse_scores.std())


scores = cross_val_score(lin_model, train_data, train_labels, scoring = "neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()