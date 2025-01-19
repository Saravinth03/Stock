import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

# 1. Load or create the dataset (expanded example)
data = {
    'PE_ratio': [15.0, 30.5, 8.9, 12.7, 25.3, 20.0, 35.0, 10.5, 22.0, 18.5, 14.7, 28.9, 16.8, 21.3, 9.0, 11.5],
    'PB_ratio': [1.2, 4.5, 0.8, 1.1, 2.7, 1.5, 3.0, 1.8, 2.2, 1.9, 1.3, 3.2, 1.4, 2.1, 0.9, 1.6],
    'DE_ratio': [0.5, 1.8, 0.3, 0.9, 2.0, 0.7, 1.2, 0.6, 1.0, 0.8, 0.4, 1.5, 0.5, 1.1, 0.3, 0.7],
    'EPS_growth': [10.0, 5.5, 12.5, 8.0, 7.3, 11.0, 4.0, 9.0, 6.5, 8.3, 10.2, 7.8, 9.4, 6.0, 11.5, 7.2],
    'undervalued': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # 1 = undervalued, 0 = not undervalued
}
df = pd.DataFrame(data)

# 2. Split the data into features and target
X = df[['PE_ratio', 'PB_ratio', 'DE_ratio', 'EPS_growth']]  # Features
y = df['undervalued']  # Target

# 3. Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5. Build the neural network model using Keras
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# 8. Evaluate the model on the test set
y_pred = (model.predict(X_test) > 0.5).astype('int32')
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. User input for predictions with error handling
def get_user_input():
    try:
        print("Please provide the stock details:")
        PE_ratio = float(input("P/E ratio: "))
        PB_ratio = float(input("P/B ratio: "))
        DE_ratio = float(input("D/E ratio: "))
        EPS_growth = float(input("EPS growth (%): "))
        
        user_input = np.array([[PE_ratio, PB_ratio, DE_ratio, EPS_growth]])
        user_input_scaled = scaler.transform(user_input)  # Scale the input
        return user_input_scaled
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return None

# 10. Predict whether the stock is undervalued based on user input
user_input = get_user_input()
if user_input is not None:
    predicted_value = model.predict(user_input)
    if predicted_value > 0.5:
        print("The stock is likely undervalued.")
    else:
        print("The stock is not likely undervalued.")