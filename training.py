import face_recognition
import pickle
import os

Encodings = []
Names = []

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
        print(path)
print(Names)

# save training data
with open('train.pkl','wb') as f:
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)
    print('training finished!')
    f.close()

