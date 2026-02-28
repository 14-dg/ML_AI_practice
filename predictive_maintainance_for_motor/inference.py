import numpy as np
import tensorflow as tf

# 1. Laden des TFLite-Modells und Zuweisung von Ressourcen
interpreter = tf.lite.Interpreter(model_path="motor_monitor.tflite")
interpreter.allocate_tensors()

# 2. Details zu Input und Output abrufen
# Das Modell muss wissen, in welches "Fach" es die Daten legen soll
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Einen neuen Sensor-Datenpunkt simulieren (z.B. "Normaler Betrieb")
# Wichtig: Die Form muss exakt (1, 100, 1) sein -> (Batch-Größe, Länge, Kanäle)
t = np.linspace(0, 1, 100)
new_sensor_data = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, 100)
new_sensor_data = new_sensor_data.astype(np.float32).reshape(1, 100, 1)

# 4. Daten in den Input-Tensor kopieren
interpreter.set_tensor(input_details[0]['index'], new_sensor_data)

# 5. Die Inferenz (Berechnung) ausführen
interpreter.invoke()

# 6. Das Ergebnis aus dem Output-Tensor abrufen
output_data = interpreter.get_tensor(output_details[0]['index'])
prediction = np.argmax(output_data)

# 7. Ergebnis interpretieren
classes = ["Normaler Betrieb", "Lagerschaden", "Kritischer Zustand"]
print(f"Modell-Vorhersage: {classes[prediction]} (Konfidenz: {output_data[0][prediction]:.2%})")