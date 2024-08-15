import pandas as pd 
from sklearn.svm import SVR 
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
# Load the data from the Excel sheet 
file_path = '/content/Crops_Data.xlsx' 
df = pd.read_excel(file_path) 
# Split the data into input and output variables 
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 
# Split the data into training and testing sets 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=0) 
# Scale the input variables 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 
# SVR algorithm 
regressor_svr = SVR(kernel='rbf') 
regressor_svr.fit(X_train, y_train) 
# Predict the output variable for the test set 
y_pred_svr = regressor_svr.predict(X_test) 
# Evaluate the performance of the model 
mse_svr = mean_squared_error(y_test, y_pred_svr) 
r2_svr = r2_score(y_test, y_pred_svr) 
# Get the accuracy of the model 
accuracy_svr = regressor_svr.score(X_test, y_test) 
# Print the performance metrics and accuracy 
print('Mean Squared Error (SVR):', mse_svr) 
print('R^2 (SVR):', r2_svr) 
print('Accuracy (SVR):', (accuracy_svr)*100) 
