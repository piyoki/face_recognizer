import face_recognition
import cv2
import numpy as np
import time

donFace = face_recognition.load_image_file('./images/known/Donald Trump.jpg')
donEncode = face_recognition.face_encodings(donFace)[0]  # return an array

nancyFace = face_recognition.load_image_file('./images/known/Nancy Pelosi.jpg')
nancyEncode = face_recognition.face_encodings(nancyFace)[0]  # return an array

Encodings = [donEncode,nancyEncode]
Names = ['Donald Trump','Nancy Pelosi']

font = cv2.FONT_HERSHEY_SIMPLEX
testImage = face_recognition.load_image_file('./images/unknown/u11.jpg')
blk = np.zeros(testImage.shape, np.uint8)
facePosition = face_recognition.face_locations(testImage)
allEncodings = face_recognition.face_encodings(testImage,facePosition)

testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
fps_count = 0

timer=cv2.getTickCount()
for (top,right,bottom,left), face_encoding in zip(facePosition, allEncodings):
    name = 'Unknown Person'
    matches = face_recognition.compare_faces(Encodings,face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        name = Names[first_match_index]
    cv2.rectangle(testImage, (right, top), (left, bottom), (0, 0, 255), 2)
    cv2.putText(testImage, name, (left, top-6), font, 0.75, (255,255,255), 2)
    cv2.rectangle(blk, (right, top), (left, bottom), (0, 0, 255), cv2.FILLED)

# Adding overlay
out = cv2.addWeighted(testImage, 1, blk, 0.3, 1)
# Display the resulting image
fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
cv2.putText(out, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
cv2.putText(out, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
cv2.imshow('myWindow',out)
cv2.moveWindow('myWindow',0,0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

