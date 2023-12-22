import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


model = MobileNetV2(weights='imagenet')

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    input_frame = cv2.resize(frame, (224, 224))

   
    input_array = preprocess_input(np.expand_dims(input_frame, axis=0))

   
    predictions = model.predict(input_array)

    
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    captions = []
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        text = "{}: {:.2f}%".format(label, score * 100)
        captions.append(text)
        cv2.putText(frame, text, (10, (i + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    caption_text = ", ".join(captions)
    cv2.putText(frame, caption_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

   
    cv2.imshow('Live Image Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
