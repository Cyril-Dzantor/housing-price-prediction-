from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(train_data, train_labels)

tree_predicted_val = tree_model.predict(test_data[:5])
tree_mse = mean_squared_error(actual_val, tree_predicted_val)
tree_rmse = root_mean_squared_error(actual_val, tree_predicted_val)
tree_mae = mean_absolute_error(actual_val, tree_predicted_val)
print(f"The mean squared error : {tree_mse}")
print(f"The root mean squared error : {tree_rmse}")
print(f"The mean absolute error : {tree_mae}")

scores = cross_val_score(tree_model, test_data,test_labels, scoring = "neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores : ",tree_rmse_scores)
print("Mean : ", tree_rmse_scores.mean())
print("Standard deviation : ", tree_rmse_scores.std())