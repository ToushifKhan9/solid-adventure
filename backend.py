import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
products_df = pd.read_csv('olist_products_dataset.csv')
orders_df = pd.read_csv('olist_order_items_dataset.csv')

df_train = pd.merge(orders_df, products_df, on='product_id', how='inner')

print("Data loaded and merged successfully!")

import pandas as pd
import numpy as np
import os

if 'shipping_limit_date' in df_train.columns:
    df_train['shipping_limit_date'] = pd.to_datetime(df_train['shipping_limit_date'])
    time_col = 'shipping_limit_date'
else:
    print("Warning: We need a date column to extract temporal features!")

df_train['day_of_week'] = df_train[time_col].dt.dayofweek
df_train['is_weekend'] = df_train['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df_train['month'] = df_train[time_col].dt.month

category_avg_price = df_train.groupby('product_category_name')['price'].transform('mean')
df_train['category_avg_price'] = category_avg_price

df_train['price_ratio'] = df_train['price'] / df_train['category_avg_price']

df_train['total_cost'] = df_train['price'] + df_train['freight_value']

df_train = df_train.dropna(subset=['price', 'freight_value', 'product_category_name'])

ml_features = ['price', 'freight_value', 'total_cost', 'category_avg_price',
               'price_ratio', 'day_of_week', 'is_weekend', 'month']

print("--- Feature Engineering Complete! ---")
print(f"Total clean rows ready for training: {len(df_train)}")
print("\nFirst 5 rows of our ML features:")
print(df_train[ml_features].head())

import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

daily_demand = df_train.groupby(['product_category_name', df_train[time_col].dt.date]).agg(
    daily_sales_volume=('price', 'count'), 
    avg_daily_price=('price', 'mean'),
    avg_freight=('freight_value', 'mean'),
    category_avg_price=('category_avg_price', 'first'),
    is_weekend=('is_weekend', 'first'),
    month=('month', 'first')
).reset_index()

daily_demand['price_ratio'] = daily_demand['avg_daily_price'] / daily_demand['category_avg_price']

daily_demand = daily_demand.dropna()

X = daily_demand[['avg_daily_price', 'avg_freight', 'category_avg_price', 'price_ratio', 'is_weekend', 'month']]
y = daily_demand['daily_sales_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_path = 'pricing_model.keras'

if os.path.exists(model_path):
    print("Pre-trained model found! Loading model instantly...")
    model = load_model(model_path)
else:
    print("No pre-trained model found. Starting training...")
    
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=['mae'])

    print("Starting training...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    print("Training Complete!")
    model.save(model_path)
    
import numpy as np

def optimize_price(category_avg_price, avg_freight, is_weekend, month, price_min=10, price_max=150):
    """
    Simulates different prices to find the optimal price point that maximizes revenue.
    """
    test_prices = np.arange(price_min, price_max, 5) 
    projected_revenues = []
    predicted_demands = []

    for price in test_prices:
        simulated_ratio = price / category_avg_price

        input_data = np.array([[price, avg_freight, category_avg_price, simulated_ratio, is_weekend, month]])

        input_scaled = scaler.transform(input_data)

        pred_demand = model.predict(input_scaled, verbose=0)[0][0]

        pred_demand = max(0, pred_demand)

        revenue = price * pred_demand

        predicted_demands.append(pred_demand)
        projected_revenues.append(revenue)

    optimal_index = np.argmax(projected_revenues)
    optimal_price = test_prices[optimal_index]
    max_revenue = projected_revenues[optimal_index]

    ##fig, ax1 = plt.subplots(figsize=(10, 5))
    #ax1.plot(test_prices, projected_revenues, color=color, marker='o', linewidth=2)
    #ax1.tick_params(axis='y', labelcolor=color)
    #ax1.axvline(x=optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price}')
    #ax1.legend(loc='upper left')

    #ax2 = ax1.twinx()
    #color = 'tab:blue'
    #ax2.set_ylabel('Predicted Demand (Units)', color=color)
    #ax2.plot(test_prices, predicted_demands, color=color, linestyle='--')
    #ax2.tick_params(axis='y', labelcolor=color)

    #plt.title(f'Dynamic Pricing Optimization\nMax Revenue: ${max_revenue:.2f} at ${optimal_price}')
    #fig.tight_layout()
    #plt.show()

    return optimal_price, max_revenue

print("Running Price Optimization Engine...")
optimal_p, max_rev = optimize_price(category_avg_price=50.0, avg_freight=15.0, is_weekend=0, month=11)
final_results = {
    "status": "success",
    "optimal_price": float(optimal_p), 
    "max_revenue": float(max_rev)
}

print(final_results)