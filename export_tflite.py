# export_tflite.py
import tensorflow as tf

def convert_to_tflite(keras_model_path='final_mnist.h5', tflite_output='model.tflite'):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # optional quantization for size & speed: uncomment for post-training quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # (for integer quantization you'd supply a representative dataset)
    tflite_model = converter.convert()
    with open(tflite_output, 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite model to", tflite_output)

if __name__ == "__main__":
    convert_to_tflite()
