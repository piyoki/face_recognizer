import face_recognition
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.5

image = face_recognition.load_image_file('./images/unknown/u3.jpg')
image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
face_locations = face_recognition.face_locations(image)  # find faces
image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
blk = np.zeros(image.shape, np.uint8)

count = 0
for face in face_locations:
    row1,col1,row2,col2 = face
    cv2.rectangle(image, (col1, row1), (col2, row2), (0, 255, 0), 2)
    cv2.rectangle(blk,(col1,row1),(col2,row2),(0,255,0),cv2.FILLED)
    count += 1

print('[INFO] {} People Detected'.format(count))
# Adding overlay
out = cv2.addWeighted(image, 1, blk, 0.3, 1)
cv2.putText(out, 'Total: {}'.format(str(count)), (11, 25), font, 0.5, (32, 32, 32), 4, cv2.LINE_AA)
cv2.putText(out, 'Total: {}'.format(str(count)), (10, 25), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
# Display the resulting image
cv2.imshow('myWindow',out)
cv2.moveWindow('myWindow',0,0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
