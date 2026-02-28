import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATENGENERIERUNG (Physikalische Simulation)
# ==========================================
def generate_vibration_data(n_samples=1000, seq_len=100):
    X, y = [], []
    t = np.linspace(0, 1, seq_len)
    
    for _ in range(n_samples):
        # Klasse 0: Normaler Betrieb (Sauberer Sinus + wenig Rauschen)
        X.append(np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, seq_len))
        y.append(0)
        
        # Klasse 1: Lagerschaden (Zusätzliche Hochfrequenz + Spikes)
        X.append(np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.2, seq_len))
        y.append(1)
        
        # Klasse 2: Kritischer Zustand (Starkes Rauschen / Unregelmäßigkeit)
        X.append(np.random.normal(0, 0.8, seq_len))
        y.append(2)
        
    X = np.array(X).reshape(-1, seq_len, 1) # Format für Conv1D: (Samples, Zeitschritte, Features)
    y = np.array(y)
    return X, y

# Daten erzeugen
X, y = generate_vibration_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. MODELL-ARCHITEKTUR (1D-CNN)
# ==========================================
# Ein 1D-CNN ist ideal für Zeitreihen/Sensordaten
model = models.Sequential([
    layers.Input(shape=(100, 1)),
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(), # Reduziert Daten für den Klassifizierer
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2), # Verhindert Overfitting
    layers.Dense(3, activation='softmax') # 3 Klassen Ausgang
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 3. TRAINING
# ==========================================
print("Starte Training...")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# ==========================================
# 4. AUSWERTUNG & EDGE AI EXPORT
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Genauigkeit: {test_acc:.4f}")

# WICHTIG FÜR RHEINMETALL: Konvertierung zu TensorFlow Lite (für Mikrocontroller/Edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('motor_monitor.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modell als 'motor_monitor.tflite' für Edge-Hardware gespeichert.")

# ==========================================
# 5. VISUALISIERUNG (Optional für dein Verständnis)
# ==========================================
plt.plot(X[0], label="Normal")
plt.plot(X[1], label="Lagerschaden")
plt.plot(X[2], label="Kritisch")
plt.legend()
plt.title("Sensordaten-Muster")
plt.show()