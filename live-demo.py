import cv2
import face_recognition
import numpy as np
import pickle

# load training data
with open('train.pkl','rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

# reconizer
cap = cv2.VideoCapture('./test.mp4')

font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.25

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('out.mp4',fourcc,10.0,(640,480))

while cap.isOpened():
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if ret == False:
        print("Camera read Error")
        break
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Resize frame of video to 1/4 * size
    rgb = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
    # Find all the faces and face encodings in the frame of video
    facePositions = face_recognition.face_locations(rgb)
    allEncodings = face_recognition.face_encodings(rgb,facePositions)
    img = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    blk = np.zeros(img.shape, np.uint8)

    people = []
    count = 0
    # Find coordinates & draw bboxes
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

        cv2.rectangle(img, (right, top), (left, bottom), color, 2)
        cv2.putText(img, name, (left, top - 6), font, 0.5, (255, 255, 255), 2)
        cv2.rectangle(blk, (right, top), (left, bottom), color, cv2.FILLED)

    print('[INFO] {} People Detected. Recognized people: {}'.format(count,people))
    # Adding overlay
    img = cv2.addWeighted(img, 1, blk, 0.4, 1)
    # Display the resulting image
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imshow('Face Detector', img)
    output.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
