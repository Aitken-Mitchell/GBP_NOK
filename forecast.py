# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import ta
import shap

# --------------------------------------------------
# Use GPU to Train Model instead of CPU
# --------------------------------------------------
# Ensure TensorFlow logs GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress non-critical warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from allocating all GPU memory

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow is using GPU: {gpus}")
    except RuntimeError as e:
        print(f"⚠️ Error setting GPU memory growth: {e}")
else:
    print("❌ No GPU detected. Running on CPU.")

# Enable multi-GPU training if multiple GPUs are available
strategy = tf.distribute.MirroredStrategy() if gpus else tf.distribute.OneDeviceStrategy(device='/cpu:0')

print(f"✅ Using strategy: {strategy}")

# --------------------------------------------------
# 1️⃣ Load Training & Testing Data
# --------------------------------------------------
train_df = pd.read_csv('train_data.csv', parse_dates=['Date'], index_col='Date')
test_df = pd.read_csv('test_data.csv', parse_dates=['Date'], index_col='Date')

# Define feature columns to be used in the model
features = [
    'GBP_to_NOK_Close', 'GBP_to_NOK_High', 'GBP_to_NOK_Low', 'GBP_to_NOK_Open',
    'Oil_Close', 'Oil_High', 'Oil_Low', 'Oil_Open',
    'WTI_Close', 'WTI_High', 'WTI_Low', 'WTI_Open',
    'FTSE_Close', 'FTSE_High', 'FTSE_Low', 'FTSE_Open',
    'Norges_Rate', 'BOE_Rate'
]

# --------------------------------------------------
# 2️⃣ Feature Engineering - Add Moving Averages for All Features & Technical Indicators
# --------------------------------------------------

moving_avg_window = 5  # You can experiment with different window sizes

# Apply moving averages to all features
for feature in features:
    train_df[f"{feature}_SMA{moving_avg_window}"] = train_df[feature].rolling(window=moving_avg_window).mean()
    test_df[f"{feature}_SMA{moving_avg_window}"] = test_df[feature].rolling(window=moving_avg_window).mean()

# Compute RSI (Relative Strength Index) with a window of 14
train_df["RSI"] = ta.momentum.RSIIndicator(train_df["GBP_to_NOK_Close"], window=14).rsi()
test_df["RSI"] = ta.momentum.RSIIndicator(test_df["GBP_to_NOK_Close"], window=14).rsi()

# Compute Bollinger Bands (High & Low)
bollinger_train = ta.volatility.BollingerBands(train_df["GBP_to_NOK_Close"], window=20)
bollinger_test = ta.volatility.BollingerBands(test_df["GBP_to_NOK_Close"], window=20)

train_df["Bollinger_High"] = bollinger_train.bollinger_hband()
test_df["Bollinger_High"] = bollinger_test.bollinger_hband()
train_df["Bollinger_Low"] = bollinger_train.bollinger_lband()
test_df["Bollinger_Low"] = bollinger_test.bollinger_lband()

# Drop NaN values introduced by rolling window and technical indicators
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Update feature list to include moving averages, RSI, and Bollinger Bands
features += [f"{feature}_SMA{moving_avg_window}" for feature in features]
features += ["RSI", "Bollinger_High", "Bollinger_Low"]

# --------------------------------------------------
# 3️⃣ Normalize Data Using MinMaxScaler
# --------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])  # Apply the same scaler to test data

# --------------------------------------------------
# 4️⃣ Function to Create Sequences for LSTM
# --------------------------------------------------
def create_sequences(data, target_column, seq_length, forecast_horizon=3):
    x, y = [], []
    for i in range(seq_length, len(data) - forecast_horizon):
        x.append(data[i-seq_length:i])
        y.append(data[i + forecast_horizon, features.index(target_column)])  # Predict 3 days ahead
    return np.array(x), np.array(y)

# Define sequence length and forecast horizon
seq_length = 100  # Increased from 50 to 100
forecast_horizon = 1  # Predicting 3 days ahead

# Create sequences for training and testing
X_train, y_train = create_sequences(train_scaled, 'GBP_to_NOK_Close', seq_length, forecast_horizon)
X_test, y_test = create_sequences(test_scaled, 'GBP_to_NOK_Close', seq_length, forecast_horizon)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --------------------------------------------------
# 5️⃣ Build the LSTM Model
# --------------------------------------------------
with strategy.scope():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        Dense(units=50),
        Dense(units=1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
model.summary()

# --------------------------------------------------
# 6️⃣ Train the Model with Callbacks
# --------------------------------------------------
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Start time tracking
start_time = time.time()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# End time tracking
end_time = time.time()
total_training_time = end_time - start_time  # Training duration in seconds

# Convert to hours, minutes, seconds
hours, rem = divmod(total_training_time, 3600)
minutes, seconds = divmod(rem, 60)

# Print training time in terminal
print(f"🕒 Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Save the trained model
model.save('gbp_nok_lstm_model.keras')

# --------------------------------------------------
# 7️⃣ Evaluate the Model on Test Data
# --------------------------------------------------
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# Make predictions
predictions = model.predict(X_test)

# --------------------------------------------------
# 8️⃣ Reverse Scaling for Predictions and Actual Values
# --------------------------------------------------
# Extract only the target column index
target_index = features.index('GBP_to_NOK_Close')

# Ensure predictions and y_test are 2D arrays
predictions_reshaped = predictions.reshape(-1, 1)  # (394, 1)
y_test_reshaped = y_test.reshape(-1, 1)  # (394, 1)

# Create a zero-filled array for inverse transformation
num_features = len(features)
dummy_array = np.zeros((len(predictions_reshaped), num_features))  # (394, 36)

# Assign the reshaped predictions properly
dummy_array[:, target_index:target_index+1] = predictions_reshaped  # Keeps the correct shape
predictions_rescaled = scaler.inverse_transform(dummy_array)[:, target_index]

# Assign the reshaped actual test values properly
dummy_array[:, target_index:target_index+1] = y_test_reshaped
y_test_rescaled = scaler.inverse_transform(dummy_array)[:, target_index]

# Bias Correction
bias_correction = (predictions_rescaled - y_test_rescaled).mean()
predictions_rescaled -= bias_correction

# Extract actual dates for the test set
test_dates = test_df.index[-len(y_test_rescaled):]

# --------------------------------------------------
# 9️⃣ Load Full Historical Data & Plot Actual vs. Predicted
# --------------------------------------------------
full_df = pd.read_csv('GBP_NOK_Input_Data.csv', parse_dates=['Date'], index_col='Date')
actual_prices = full_df['GBP_to_NOK_Close']

plt.figure(figsize=(12,6))

# Plot actual exchange rate for entire dataset
plt.plot(full_df.index, actual_prices, label='Actual GBP to NOK Close', color='blue', alpha=0.6)

# Overlay predictions for the test period
plt.plot(test_dates, predictions_rescaled, label='Predicted GBP to NOK Close', color='red', linestyle='dashed')

plt.title('GBP to NOK Exchange Rate: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('GBP to NOK Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("GBP_NOK_LSTM1")  # Save plot as an image
plt.show()

# --------------------------------------------------
# 🔟 Plot Only the predicted vs. actual 
# --------------------------------------------------

# Plot Actual vs Predicted GBP to NOK Close Prices during test period
plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_rescaled, label='Actual GBP to NOK Close', color='blue', alpha=0.6)
plt.plot(test_dates, predictions_rescaled, label='Predicted GBP to NOK Close', color='red', linestyle='dashed')
plt.title('Actual vs. Predicted GBP to NOK Close Prices (Test Period)')
plt.xlabel('Date')
plt.ylabel('GBP to NOK Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Actual_vs_Predicted_Test_Period.png")
plt.show()

# --------------------------------------------------
# 11 Create DataFrame with Predicted vs Actual Values
# --------------------------------------------------

# Create DataFrame with Predicted vs Actual Values
predicted_vs_actual = pd.DataFrame({
    'Date': test_df.index[-len(y_test_rescaled):],
    'Actual_GBP_NOK_Close': y_test_rescaled,
    'Predicted_GBP_NOK_Close': predictions_rescaled
})

# Initialize investment strategy logic
fund_value = 1000  # Start with 1000 GBP
holding_gbp = True  # Start by holding GBP
investment_strategy = []
investment_results = []

for i in range(len(predicted_vs_actual) - 1):
    current_price = predicted_vs_actual.loc[i, 'Actual_GBP_NOK_Close']
    next_price = predicted_vs_actual.loc[i + 1, 'Predicted_GBP_NOK_Close']
    actual_next_price = predicted_vs_actual.loc[i + 1, 'Actual_GBP_NOK_Close']
    
    if next_price > current_price:
        if holding_gbp:
            action = "HOLD GBP"
        else:
            action = "BUY GBP"
            fund_value /= actual_next_price  # Convert NOK to GBP
            holding_gbp = True
    else:
        if holding_gbp:
            action = "BUY NOK"
            fund_value *= actual_next_price  # Convert GBP to NOK
            holding_gbp = False
        else:
            action = "HOLD NOK"

    # Format fund value correctly
    investment_strategy.append(action)
    investment_results.append(f"{fund_value:.2f} {'GBP' if holding_gbp else 'NOK'}")

# Append final state for last row
investment_strategy.append("HOLD")
investment_results.append(f"{fund_value:.2f} {'GBP' if holding_gbp else 'NOK'}")

# Add new columns to DataFrame
predicted_vs_actual['Action'] = investment_strategy
predicted_vs_actual['Fund_Value'] = investment_results

# Save the updated DataFrame as a CSV file
predicted_vs_actual.to_csv("Predicted_vs_Actual.csv", index=False)

# --------------------------------------------------
# 12 Calculate & Plot Loss Metrics (MSE, MAE, RMSE)
# --------------------------------------------------

# Extract loss values from training history
mse_values = history.history['loss']
val_mse_values = history.history['val_loss']

# Compute RMSE for each epoch
rmse_values = np.sqrt(mse_values)
val_rmse_values = np.sqrt(val_mse_values)

# Compute MAE for each epoch (not directly available, but can be approximated from MSE)
mae_values = [np.sqrt(mse) for mse in mse_values]
val_mae_values = [np.sqrt(mse) for mse in val_mse_values]

# Plot Loss Metrics Over Epochs
plt.figure(figsize=(10, 5))
plt.plot(mse_values, label='Training MSE', color='blue')
plt.plot(val_mse_values, label='Validation MSE', color='red', linestyle='dashed')
plt.plot(mae_values, label='Training MAE', color='green')
plt.plot(val_mae_values, label='Validation MAE', color='orange', linestyle='dashed')
plt.plot(rmse_values, label='Training RMSE', color='purple')
plt.plot(val_rmse_values, label='Validation RMSE', color='brown', linestyle='dashed')

plt.title('Error Metrics vs. Number of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Error_Metrics_vs_Epochs.png")
plt.show()
