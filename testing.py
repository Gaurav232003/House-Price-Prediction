import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Model training and testing
def train_and_test_model():
    # Load the training data from CSV
    train_df = pd.read_csv('data.csv')

    # Fill missing values in the 'bedrooms' column with the median value of the existing 'bedrooms' data
    median_bedrooms = np.floor(train_df.bedrooms.median())
    train_df.bedrooms = train_df.bedrooms.fillna(median_bedrooms)

    # Fill missing values in other columns with median values
    median_sea_facing = train_df.sea_facing.median()
    train_df.sea_facing = train_df.sea_facing.fillna(median_sea_facing)

    median_nearby_schools = np.floor(train_df.nearby_schools.median())
    train_df.nearby_schools = train_df.nearby_schools.fillna(median_nearby_schools)

    median_nearby_malls = np.floor(train_df.nearby_malls.median())
    train_df.nearby_malls = train_df.nearby_malls.fillna(median_nearby_malls)

    median_avg_cost_nearby_houses = np.floor(train_df.avg_cost_nearby_houses.median())
    train_df.avg_cost_nearby_houses = train_df.avg_cost_nearby_houses.fillna(median_avg_cost_nearby_houses)

    median_floor = np.floor(train_df.floor.median())
    train_df.floor = train_df.floor.fillna(median_floor)

    # Define features and target variable for training
    features = ['area', 'bedrooms', 'age', 'sea_facing', 'nearby_schools', 'nearby_malls', 
                'avg_cost_nearby_houses', 'floor']
    target = 'price'

    # Convert boolean 'sea_facing' column to integers (True: 1, False: 0)
    train_df['sea_facing'] = train_df['sea_facing'].astype(int)

    # Convert categorical 'environmental_risk' column to one-hot encoded columns
    train_df = pd.get_dummies(train_df, columns=['environmental_risk'])

    # Define the independent variables (features) and target variable
    X_train = train_df[features]
    y_train = train_df[target]

    # Create a linear regression model
    reg = LinearRegression()

    # Fit the model
    reg.fit(X_train, y_train)

    # Load the test data from CSV
    test_df = pd.read_csv('test.csv')

    # Define features for testing
    features = ['area', 'bedrooms', 'age', 'sea_facing', 'nearby_schools', 'nearby_malls', 
                'avg_cost_nearby_houses', 'floor']

    # Extract features from test data
    X_test = test_df[features]

    # Extract actual prices from test data
    y_test = test_df['price']

    # Predict using the trained model
    y_pred = reg.predict(X_test)

    # Calculate mean squared error (MSE) and mean absolute error (MAE)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Print MSE and MAE
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    # Calculate accuracy percentage
    tolerance = 0.05  # Tolerance level for percentage accuracy
    accurate_predictions = np.abs(y_pred - y_test) / y_test <= tolerance
    accuracy_percentage = (np.sum(accurate_predictions) / len(y_test)) * 100

    # Print accuracy percentage
    print("Accuracy Percentage:", accuracy_percentage)

# Train and test the model
train_and_test_model()
