import cv2
import face_recognition
import numpy as np
import os

Encodings = []
Names = []
font = cv2.FONT_HERSHEY_SIMPLEX

# trainer
image_dir = './images/known'
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root,file)
        name = os.path.splitext(file)[0]
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]
        Encodings.append(encoding)
        Names.append(name)
print(Names)

# reconizer
timer = cv2.getTickCount()
testImage = face_recognition.load_image_file('./images/unknown/u1.jpg')
facePositions = face_recognition.face_locations(testImage)
allEncodings = face_recognition.face_encodings(testImage,facePositions)
testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
blk = np.zeros(testImage.shape, np.uint8)

for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
    name = 'Unknown Person'
    matches = face_recognition.compare_faces(Encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        color = (0,255,0)
        name = Names[first_match_index]
    else:
        color = (0,0,255)

    cv2.rectangle(testImage, (right, top), (left, bottom), color, 2)
    cv2.putText(testImage, name, (left, top - 6), font, 0.75, (255, 255, 255), 2)
    cv2.rectangle(blk, (right, top), (left, bottom), color, cv2.FILLED)

# Adding overlay
out = cv2.addWeighted(testImage, 1, blk, 0.4, 1)
# Display the resulting image
fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
cv2.putText(out, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
cv2.putText(out, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
cv2.imshow('Face Detector', out)
cv2.moveWindow('Face Detector', 0, 0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
