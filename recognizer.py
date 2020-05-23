import face_recognition
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

image = face_recognition.load_image_file('./images/unknown/u3.jpg')
face_locations = face_recognition.face_locations(image)  # find faces
print(face_locations)
image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
blk = np.zeros(image.shape, np.uint8)

timer=cv2.getTickCount()
for face in face_locations:
    row1,col1,row2,col2 = face
    cv2.rectangle(image, (col1, row1), (col2, row2), (0, 0, 255), 2)
    cv2.rectangle(blk,(col1,row1),(col2,row2),(0,0,255),cv2.FILLED)

# Adding overlay
out = cv2.addWeighted(image, 1, blk, 0.3, 1)
# Display the resulting image
fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
cv2.putText(out, "FPS: " + str(int(fps)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
cv2.putText(out, "FPS: " + str(int(fps)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
cv2.imshow('myWindow',out)
cv2.moveWindow('myWindow',0,0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()