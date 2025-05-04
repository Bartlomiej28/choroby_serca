
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = pd.read_csv('cardio_train.csv', sep=';')

data['age'] = data['age'] / 365
data['height'] = data['height'] / 100

new_data = data.drop(['id'], axis=1)

correlation_matrix = new_data.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Macierz korelacji cech")
plt.show()

print("ap_hi:", np.min(new_data['ap_hi']), np.max(new_data['ap_hi']), np.mean(new_data['ap_hi']))
print("ap_lo:", np.min(new_data['ap_lo']), np.max(new_data['ap_lo']), np.mean(new_data['ap_lo']))

data_filtered = new_data[(new_data['ap_hi'] >= 50) & (new_data['ap_hi'] <= 250)]
data_filtered = data_filtered[(data_filtered['ap_lo'] >= 30) & (data_filtered['ap_lo'] <= 180)]

print(f"Przed filtracją: {new_data.shape[0]} próbek")
print(f"Po filtracji: {data_filtered.shape[0]} próbek")
print(new_data.iloc[0:10])

X = data_filtered.drop('cardio', axis=1)
y = data_filtered['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])



model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train_scaled, y_train, epochs=30, batch_size=32)

print("\nPredykcje dla 10 próbek testowych:")
for i in range(10):
    sample = X_test_scaled[i].reshape(1, -1)
    pred = model.predict(sample, verbose=0)
    print(f"Próbka {i+1}: Prawdziwa wartość: {y_test.iloc[i]}, Predykcja: {int(np.round(pred))}")

