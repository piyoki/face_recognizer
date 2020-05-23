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
cap = cv2.VideoCapture('./test.webm')
cap.set(3,640)
cap.set(4,480)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('out.mp4',fourcc,10.0,(640,480))

while cap.isOpened():
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if ret == False:
        print("Camera read Error")
        break
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(640,480))
    # Find all the faces and face encodings in the frame of video
    facePositions = face_recognition.face_locations(rgb)
    allEncodings = face_recognition.face_encodings(rgb,facePositions)
    img = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    blk = np.zeros(img.shape, np.uint8)

    # Find coordinates & draw bboxes
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            color = (0,255,0)
            name = Names[first_match_index]
        else:
            color = (0,0,255)

        cv2.rectangle(img, (right, top), (left, bottom), color, 2)
        cv2.putText(img, name, (left, top - 6), font, 0.75, (255, 255, 255), 2)
        cv2.rectangle(blk, (right, top), (left, bottom), color, cv2.FILLED)

    # Adding overlay
    img = cv2.addWeighted(img, 1, blk, 0.4, 1)
    # Display the resulting image
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imshow('Face Detector', img)
    cv2.moveWindow('Face Detector', 0, 0)
    output.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()import cv2
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
cap = cv2.VideoCapture('./test.webm')
cap.set(3,640)
cap.set(4,480)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('out.mp4',fourcc,10.0,(640,480))

while cap.isOpened():
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if ret == False:
        print("Camera read Error")
        break
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(640,480))
    # Find all the faces and face encodings in the frame of video
    facePositions = face_recognition.face_locations(rgb)
    allEncodings = face_recognition.face_encodings(rgb,facePositions)
    img = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    blk = np.zeros(img.shape, np.uint8)

    # Find coordinates & draw bboxes
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            color = (0,255,0)
            name = Names[first_match_index]
        else:
            color = (0,0,255)

        cv2.rectangle(img, (right, top), (left, bottom), color, 2)
        cv2.putText(img, name, (left, top - 6), font, 0.75, (255, 255, 255), 2)
        cv2.rectangle(blk, (right, top), (left, bottom), color, cv2.FILLED)

    # Adding overlay
    img = cv2.addWeighted(img, 1, blk, 0.4, 1)
    # Display the resulting image
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imshow('Face Detector', img)
    cv2.moveWindow('Face Detector', 0, 0)
    output.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
