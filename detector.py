import cv2
import face_recognition
import numpy as np
import pickle

# load training data
with open('train.pkl','rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.5

# reconizer
testImage = face_recognition.load_image_file('./images/unknown/u11.jpg')
testImage = cv2.resize(testImage, (0, 0), fx=scale, fy=scale)
facePositions = face_recognition.face_locations(testImage)
allEncodings = face_recognition.face_encodings(testImage,facePositions)
testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
blk = np.zeros(testImage.shape, np.uint8)

people = []
count = 0
for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
    name = 'Unknown Person'
    matches = face_recognition.compare_faces(Encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        color = (0,255,0)
        name = Names[first_match_index]
        people.append(name)

    else:
        color = (0,0,255)
    count += 1
    cv2.rectangle(testImage, (right, top), (left, bottom), color, 2)
    cv2.putText(testImage, name, (left, top - 6), font, 0.5, (255, 255, 255), 2)
    cv2.rectangle(blk, (right, top), (left, bottom), color, cv2.FILLED)

print('[INFO] {} People Detected. Recognized people: {}'.format(count,people))
# Adding overlay
out = cv2.addWeighted(testImage, 1, blk, 0.4, 1)
cv2.putText(out, 'Total: {}'.format(str(count)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
cv2.putText(out, 'Total: {}'.format(str(count)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
cv2.imshow('Face Detector', out)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

