# infer.py
import cv2
import numpy as np
import tensorflow as tf
from utils import preprocess_for_mnist, segment_digits_from_image
from tensorflow.keras.models import load_model

def load_model_path(path='best_mnist.h5'):
    return load_model(path)

def predict_single(model, img_path, show=False):
    img = cv2.imread(img_path)
    proc = preprocess_for_mnist(img)   # returns 28x28 float normalized
    x = proc.reshape(1,28,28,1).astype('float32')
    preds = model.predict(x)
    label = int(np.argmax(preds, axis=1)[0])
    conf  = float(np.max(preds))
    if show:
        cv2.imshow('Digit', cv2.resize(img, (200,200)))
        print("Predicted:", label, "conf:", conf)
        cv2.waitKey(0)
    return label, conf

def predict_multidigit(model, img_path):
    img = cv2.imread(img_path)
    crops, boxes = segment_digits_from_image(img)
    results = []
    for crop in crops:
        # preprocess each crop like MNIST
        proc = crop  # crop is thresholded already (from utils)
        proc = cv2.resize(proc, (28,28), interpolation=cv2.INTER_AREA)
        proc = proc.astype('float32')/255.0
        x = proc.reshape(1,28,28,1)
        preds = model.predict(x)
        results.append((int(np.argmax(preds)), float(np.max(preds))))
    return results

if __name__ == "__main__":
    model = load_model_path('best_mnist.h5')
    print(predict_single(model, 'samples/digit_photo.jpg', show=False))
    print(predict_multidigit(model, 'samples/multi_digits.jpg'))
