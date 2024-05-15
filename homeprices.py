import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("data.csv")

# Fill missing values in the 'bedrooms' column with the median value of the existing 'bedrooms' data
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

# Fill missing values in other columns with median values
median_sea_facing = df.sea_facing.median()
df.sea_facing = df.sea_facing.fillna(median_sea_facing)

median_nearby_schools = math.floor(df.nearby_schools.median())
df.nearby_schools = df.nearby_schools.fillna(median_nearby_schools)

median_nearby_malls = math.floor(df.nearby_malls.median())
df.nearby_malls = df.nearby_malls.fillna(median_nearby_malls)

median_avg_cost_nearby_houses = math.floor(df.avg_cost_nearby_houses.median())
df.avg_cost_nearby_houses = df.avg_cost_nearby_houses.fillna(median_avg_cost_nearby_houses)

median_floor = math.floor(df.floor.median())
df.floor = df.floor.fillna(median_floor)

# Define the features and target variable
features = ['area', 'bedrooms', 'age', 'sea_facing', 'nearby_schools', 'nearby_malls', 
            'avg_cost_nearby_houses', 'floor']
target = 'price'

# Convert boolean 'sea_facing' column to integers (True: 1, False: 0)
df['sea_facing'] = df['sea_facing'].astype(int)

# Convert categorical 'environmental_risk' column to one-hot encoded columns
df = pd.get_dummies(df, columns=['environmental_risk'])

# Define the independent variables (features) and target variable
X = df[features]
y = df[target]

# Create a linear regression model
reg = LinearRegression()

# Fit the model
reg.fit(X, y)

# Function to calculate average price by location
def average_price_by_location(location):
    avg_price = df[df['location'] == location]['price'].mean()
    return avg_price

# Example location
location = 'Suburb'

# Calculate average price in the specified location
avg_price_location = average_price_by_location(location)

print("Average price in", location, ":", avg_price_location)

# Predict the price for a new data point
new_data = [[2000, 1, 15, 1, 3, 2, 620000, 3]]  # Example data point with all features
predicted_price = reg.predict(new_data)

print("Predicted price:", predicted_price[0])

# Plotting the bar graph
plt.bar(['Average Price', 'Predicted Price'], [avg_price_location, predicted_price[0]])
plt.xlabel('Price Type')
plt.ylabel('Price')
plt.title('Comparison of Average Price and Predicted Price')
plt.show()
