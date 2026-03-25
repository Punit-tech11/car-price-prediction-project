import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (tum apna CSV bhi use kar sakte ho)
data = {
    "year": [2015, 2016, 2017, 2018, 2019],
    "present_price": [5.0, 6.5, 7.2, 8.0, 9.5],
    "kms_driven": [50000, 30000, 40000, 25000, 20000],
    "fuel_type": [0, 1, 0, 1, 0],  # 0=Petrol, 1=Diesel
    "seller_type": [0, 0, 1, 1, 0], # 0=Dealer, 1=Individual
    "transmission": [0, 1, 0, 1, 0], # 0=Manual, 1=Auto
    "owner": [0, 1, 0, 2, 0],
    "selling_price": [3.5, 4.0, 5.0, 6.5, 7.0]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['year', 'present_price', 'kms_driven', 'fuel_type', 'seller_type', 'transmission', 'owner']]
y = df['selling_price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "car_price_model.pkl")

print("✅ Model trained and saved successfully!")