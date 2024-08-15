 
import pandas as pd 
from sklearn.neural_network import MLPRegressor 
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
# Back Propagation Neural Networks algorithm 
regressor_bpnn = MLPRegressor(hidden_layer_sizes=(10, ), activation='relu', 
solver='adam', max_iter=10000) 
regressor_bpnn.fit(X_train, y_train) 
# Predict the output variable for the test set 
y_pred_bpnn = regressor_bpnn.predict(X_test) 
# Evaluate the performance of the model 
mse_bpnn = mean_squared_error(y_test, y_pred_bpnn) 
r2_bpnn = r2_score(y_test, y_pred_bpnn) 
# Print the performance metrics 
print('Mean Squared Error (BPNN):', mse_bpnn) 
print('R^2 (BPNN):', r2_bpnn) 
print('Accuracy % :', r2_bpnn*100) 
