from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import root_mean_squared_error
lin_model = LinearRegression()
lin_model.fit(train_data, train_labels)

actual_val = test_labels[:5]
predicted_val = lin_model.predict(test_data[:5])

comp_df = pd.DataFrame(data={"Actual Values":actual_val, "Predicted Values":predicted_val})
comp_df["Differences"] = comp_df["Actual Values"] - comp_df["Predicted Values"]
comp_df

lin_mse = mean_squared_error(actual_val,predicted_val)
lin_rmse = root_mean_squared_error(actual_val, predicted_val)
lin_mae = mean_absolute_error(actual_val,predicted_val)
print(f"The mean squared error : {lin_mse}")
print(f"The root mean squared error : {lin_rmse}")
print(f"The mean absolute error : {lin_mae}")