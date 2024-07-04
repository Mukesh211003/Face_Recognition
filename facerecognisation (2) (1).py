import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define path to dataset
path = '/home/lohith/Desktop/face_recognition/dataset'
images = []
imgLabels = []
myList = os.listdir(path)
print(myList)

# Load images and their labels
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    imgLabels.append(os.path.splitext(cl)[0])

print(imgLabels)

def findEncodings(images):
    encodList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodList.append(encode[0])
    return encodList

# Get encodings of known faces
encoded_faces = findEncodings(images)
print('Encoding completed')

def markAttendance(name):
    with open('/home/lohith/Desktop/face_recognition/attendance1.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        else:
            # Update attendance time if already in the list
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            updatedList = [f'{name},{dtString}' if line.startswith(name) else line.strip() for line in myDataList]
            f.seek(0)
            f.writelines('\n'.join(updatedList) + '\n')

# Define threshold for recognizing faces
FACE_DISTANCE_THRESHOLD = 0.5

# Start webcam for face recognition
webcam = cv2.VideoCapture(0)

while True:
    success, img = webcam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLocation in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encoded_faces, encodeFace, tolerance=FACE_DISTANCE_THRESHOLD)
        faceDis = face_recognition.face_distance(encoded_faces, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < FACE_DISTANCE_THRESHOLD:
            name = imgLabels[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            print("Unknown face detected")
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

