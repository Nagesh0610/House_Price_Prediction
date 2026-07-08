import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv('data.csv')

# Select features
data = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors',
             'sqft_lot', 'condition', 'sqft_above', 'sqft_basement', 'price']]

# Remove missing values
data = data.dropna()

# Remove outliers
data = data[data['price'] < 2000000]

# Features and target
X = data.drop('price', axis=1)
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Updated model trained!")