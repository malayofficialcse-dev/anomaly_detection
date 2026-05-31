import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

DATASET_PATH = "ambient_temperature_system_failure.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)
print(df.head())

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.dropna()


values = df[["value"]].astype("float32")

# Normalize values
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Convert to Tensor
data_tensor = tf.convert_to_tensor(
    scaled_values,
    dtype=tf.float32
)

input_dim = scaled_values.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))

encoder = Dense(
    encoding_dim,
    activation="relu"
)(input_layer)

decoder = Dense(
    input_dim,
    activation="sigmoid"
)(encoder)

autoencoder = Model(
    inputs=input_layer,
    outputs=decoder
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

history = autoencoder.fit(
    data_tensor,
    data_tensor,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

reconstructed = autoencoder.predict(data_tensor)

mse = np.mean(
    np.square(
        scaled_values - reconstructed
    ),
    axis=1
)

anomaly_scores = pd.Series(
    mse,
    index=df.index,
    name="anomaly_score"
)


threshold = anomaly_scores.quantile(0.99)

anomalies = anomaly_scores > threshold

print("\nThreshold:", threshold)
print("Total Anomalies:", anomalies.sum())

binary_labels = anomalies.astype(int)

precision, recall, f1_score, _ = precision_recall_fscore_support(
    binary_labels,
    binary_labels,
    average="binary"
)

print("\nEvaluation Metrics")
print("-------------------")
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1_score)

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(16, 7))

plt.plot(
    df["timestamp"],
    df["value"],
    label="Temperature"
)

plt.scatter(
    df["timestamp"][anomalies],
    df["value"][anomalies],
    color="red",
    label="Anomaly"
)

plt.title("Anomaly Detection using Autoencoder")
plt.xlabel("Time")
plt.ylabel("Temperature Value")
plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 5))

plt.hist(
    anomaly_scores,
    bins=50
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.title("Distribution of Reconstruction Errors")
plt.xlabel("Anomaly Score (MSE)")
plt.ylabel("Frequency")
plt.legend()

plt.show()

sample_size = 500

actual = scaled_values[:sample_size]
predicted = reconstructed[:sample_size]

plt.figure(figsize=(14, 6))

plt.plot(actual, label="Actual")
plt.plot(predicted, linestyle="--", label="Reconstructed")

plt.title("Actual vs Reconstructed Signals")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Value")
plt.legend()

plt.show()

results = df.copy()

results["anomaly_score"] = anomaly_scores
results["is_anomaly"] = anomalies.astype(int)

results.to_csv(
    "anomaly_detection_results.csv",
    index=False
)

print("\nResults saved successfully.")