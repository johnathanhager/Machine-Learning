from keras.models import load_model
import cv2
import numpy as np
import time

REV_CLASS_MAP = { 0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 
                 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 
                 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "Nothing"}
def mapper(val):
    return REV_CLASS_MAP[val]

model = load_model("C:/Applied-Artificial-Intelligence-ECGR-6119-001/Project-1/Model_numbers/keras_model.h5")
cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (10, 70), (300, 340), (0, 255, 0), 2)
    cv2.rectangle(frame, (330, 70), (630, 370), (255, 0, 0), 2)
    # extract the region of image within the user rectangle
    roi = frame[70:300, 10:340]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) # for pc
    # img = cv2.resize(img, (277, 277)) # for pi
    # predict the model
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    print(user_move_name)
    time.sleep(1)