# ===============================
# Step 1: Import Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# Step 2: Fetch Dataset
# ===============================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Train set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)

# ===============================
# Step 3: Preprocess Data
# ===============================
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ===============================
# Step 4: Build CNN Model
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ===============================
# Step 5: Train Model
# ===============================
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# ===============================
# Step 6: Evaluate Model
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("✅ Test accuracy:", test_acc)

# ===============================
# Step 7: Save Model
# ===============================
model.save("best_mnist.h5")
print("✅ Model saved as best_mnist.h5")

# ===============================
# Step 8: Plot Accuracy
# ===============================
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

# ===============================
# Step 9: Test Prediction (Optional)
# ===============================
# Predict the first 5 test images
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.show()
