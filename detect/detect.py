from os import makedirs
from os.path import isdir
from urllib import request

import cv2

dir = "faces/"
face_filter = cv2.CascadeClassifier('C:/Users/qkrtj/PycharmProjects/Django_openCV/detect'
                                    '/haarcascade_frontalface_default.xml')


def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_filter.detectMultiScale(gray, 1.3)

    if faces is():
        return None

    for (x, y, w, h) in faces:
        new_face_image = image[y:y + h, x:x + h]

    return new_face_image


def save(name):
    if not isdir(dir + name):
        makedirs(dir + name)

    cap = cv2.VideoCapture(0)
    cnt = 0
    ret, frame = cap.read()
    while True:
        if face_detect(frame):
            face = cv2.resize(face_detect(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            path_name = dir + name + '/user' + str(cnt) + '.jpg'
            cv2.imwrite(path_name, face)
            cv2.putText(face, str(cnt), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('face_detect', face)
        else:
            print('not found')
            pass
        if cv2.waitKey(1) == 13 or cnt == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    # def face_detect(image):
    #    # image = cv2.imread(image_url)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     faces = face_filter.detectMultiScale(gray, 1.3)
    #
    #     for (x, y, w, h) in faces:
    #         new_face_image = image[y:y + h, x:x + h]
    #     return new_face_image