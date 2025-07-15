
---

## 🧠 How It Works

1. **Face Detection**: Images are processed using OpenCV or other methods to extract face regions.
2. **Preprocessing**: Images are resized and normalized to feed into the ResNet model.
3. **Model Training**: A ResNet-based CNN is trained to classify real vs. fake faces.
4. **Evaluation**: Results are evaluated using accuracy, precision, recall, and confusion matrix.

---

## 🔍 Sample Code Snippet

```python
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights=None, input_shape=(128, 128, 3), classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
