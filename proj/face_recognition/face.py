#-*-coding:utf8-*-

import cv2
import os

FACE_DATA = './facedata'

def face_collect():
    global FACE_DATA
    folder_name = FACE_DATA

    IMG_CNT = 100
    KEY_INTERVAL = 1 # 1s

    cap = cv2.VideoCapture(0)

    cascade_file = 'haarcascade_frontalface_alt.xml'

    if not os.path.isfile(cascade_file):
        RuntimeError('file %s is not exist' % (cascade_file))
        return

    face_detector = cv2.CascadeClassifier(cascade_file)

    face_id = input('\n enter user id: ')

    print('\n Initializing face capture. Look at the camera and wait ...')

    count = 0

    while(True):
        sucess, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray is None:
            print('\n gray is null')
            break
        else:
            faces = face_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
            count += 1
            folder = "./" + folder_name + "/user_" + str(face_id)

            if not os.path.exists(folder):
                os.makedirs(folder)

            cv2.imwrite(folder + '/' + str(count) + '.jpg', gray[y: y + h, x: x + w])
            cv2.imshow('image', img)

            print(' user [%s] count [%d]' % (str(face_id), count) )

        key = cv2.waitKey(KEY_INTERVAL)

        if key & 0xFF == ord('q') or count >= IMG_CNT:
            cap.release()
            cv2.destroyAllWindows()
            break

def face_training():
    import numpy as np
    from PIL import Image

    global FACE_DATA

    print('--------face training-------\n')

    images = []

    for root, dirs, files in os.walk(FACE_DATA):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # all file
        for f in files:
            #print(os.path.join(root, f))
            images.append(os.path.join(root, f))

        # all folder
        #for d in dirs:
        #    print(os.path.join(root, d))
        pass
    for img in images:
        print(str(os.path.split(img)[0]))
        print(str(os.path.split(img)[0].split("_")[1]))
        #print(str(os.path.split(img)[-1].split("user_")[0]))

    # need add below library,
    # pip install opencv-contrib-python.
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    faceSamples = []
    ids = []

    for img in images:
        pil_img = Image.open(img).convert('L')
        img_numpy = np.array(pil_img, 'uint8')

        #id = int(os.path.split(img)[-1].split(".")[0])
        id = int(os.path.split(img)[0].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    #return faceSamples, ids
    recognizer.train(faceSamples, np.array(ids))
    recognizer.write(r'./trainer.yml')
    print("{0} faces trained. Exiting Program".format(len(np.unique(ids))))

def face_check():
    print('--------face_check-------\n')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer.yml')

    cascadePath = "haarcascade_frontalface_alt.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    idnum = 0
    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    cam = cv2.VideoCapture(0)
    minW = 0.1 * cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if idnum >= len(names):
                idnum = 'unknow'
            elif confidence < 100:
                idnum = names[idnum]
            else:
                idnum = "unknown"

            confidence = "{0}%".format(round(100 - confidence))

            # Put text on image.
            cv2.putText(img, str(idnum), (x + 5, y - 5), font, 1, (0, 0, 255), 3)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 255, 0), 3)

        cv2.imshow('camera', img)
        key = cv2.waitKey(10)

        if key & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break

    pass


if __name__ == '__main__':
    #face_collect()
    #face_training()
    #face_check()
    pass