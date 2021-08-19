import threading
from os import listdir
from os.path import isfile, join
from django.views.decorators import gzip
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from os import makedirs
from os.path import isdir
import cv2
from detect.models import Photo, User, Respone, Check, Login
import numpy as np

dir = "faces/"
# data_path = "C:/Users/qkrtj/PycharmProjects/Django_openCV/media/faces/"
face_filter = cv2.CascadeClassifier(
    'C:/Users/qkrtj/PycharmProjects/Django_openCV/detect/haarcascade_frontalface_default.xml')#absolute url


# Path for face image database

# recognizer = cv2.face.LBPHFaceRecognizer_create()


def save_image(request):
    # users = User.objects.all().order_by('datetime')
    # name = users[0]
    name = request.session['username']
    print(name)
    if not isdir("media/" + dir + str(name) + "/"):
        makedirs("media/" + dir + str(name) + "/")

    return render(request, 'save_image.html')


def train(name):
    # dir = "media/"+dir+"user" + str(re_len) + '.jpg'
    data_path = "media/faces/" + name + "/"
    file_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    data, label = [], []
    for i, files in enumerate(file_list):
        if not '.JPG' in files:  # 확장자가 jpg가 아닌 경우 무시
            continue
        image_path = data_path + file_list[i]
        # im = cv2.imread(image_path)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        data.append(np.asarray(images, dtype=np.uint8))
        label.append(i)
    label = np.asarray(label, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(data), np.asarray(label))
    # print(model)
    return model


def trains():
    data_path = 'media/faces/'
    members = User.objects.all()
    models_dirs = []

    for user in members:
        models_dirs.append(user.username)

    models = {}
    for model in models_dirs:
        result = train(model)
        if result is None:
            continue
        models[model] = result
    return models


def index(request):
    return render(request, 'index.html')


def user(request):
    if request.method == "GET":
        check = Check.objects.all()

        if str(check[0]) == "True":
            user = Login.objects.all()
            user = str(user[0])
            return render(request, 'user.html', {"user" : user})
        else:
            user = Login.objects.all()
            if user == "":
                return HttpResponse("<script>alert('Not login! Please Login!');location.href='http://127.0.0.1/'")


def register(request):
    if request.method == "GET":
        return render(request, 'register.html')

    elif request.method == "POST":
        list = []
        username = request.POST['username']
        request.session['username'] = username

        # members[0]즉 가장 최근 유저로 학습을 시킴
        members = User.objects.all()

        for user in members:
            list.append(user.username)

        if username in list:
            print("error!")
            return HttpResponse(
                '<script>alert("이미 있는 유저입니다!");location.href="http://127.0.0.1:8000/register";</script>')

        elif username not in list:
            new_user = User.objects.create(username=username)
            new_user.save()
        # members = User.objects.all().delete()
        return redirect('save_image')


# 해야할 일 사진저장 및 유저 모델 생성 및 유저 모델로 로그인
def register_image(request):
    if request.method == "GET":
        return render(request, 'register_image.html')

    elif request.method == "POST":
        #     username = request.session['username']
        #     print(username)
        images = request.FILES.getlist('images')

        for image in images:
            Photo.objects.create(images=image)

        image_all = Photo.objects.all()
        for image in image_all:
            print(image.images.url)

        return render(request, 'register_image.html', {'images': image_all})


def login(request):
    if request.method == "GET":
        users = Login.objects.all().delete()  # admin
        response = Check.objects.all().delete()
        response = Respone.objects.all().delete()
        return render(request, 'login.html')
    elif request.method == "POST":
        username = request.POST.get('user_name')
        print(len(username))

        if len(username) == 0:
            return HttpResponse(
                "<script>alert('로그인 할 유저의 이름을 입력해주세요!');location.href='http://127.0.0.1:8000/';</script>")
        else:
            login = Login.objects.create(username=username)
            login.save()

            return redirect("login")


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass


@gzip.gzip_page
def detect_image(request):
    try:  ##response가 100보다 크면 response를 0으로 만들고 login쪽으로  redirect함 ㅇㅋ?
        res = Respone.objects.all()
        res_len = len(res)
        cam = get_image()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass


class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_filter.detectMultiScale(gray, 1.3)
        login_input = Login.objects.all()
        #model = train(str(name))
        name = login_input[0]
        models = trains()
        confidence = 0

        if faces is ():  # face있는지 검출
            pass
        else:
            prediction_value = {}
            list = []  # name save
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255),
                              thickness=2)  ##아주 잘 그려짐 증명됨 개발을 할거면 image변수를 사용해야함
                image = cv2.flip(image, 0)
                cut_image = image[y:y + h, x:x + h]
                cut_image = cv2.resize(cut_image, (200, 200))
                face = cv2.cvtColor(cut_image, cv2.COLOR_BGR2GRAY)

            for key, model in models.items():
                result = model.predict(face)
                prediction_value[key] = result[1]

            print("list : ")
            print(prediction_value)
            sorted(prediction_value.items(), key=lambda x: x[1], reverse=True)
            print("after"  )
            print(prediction_value)
            for key, model in prediction_value.items():
                list.append(key)

            model_name = list[0]
            if str(model_name) == str(name):
                print("True")
                confidence = int(prediction_value[str(name)])

            else:
                print("False")
            print("model name : " + str(model_name))
            print("real name : " + str(name))
            print("confidence vali : " + str(confidence))
            print("confidence : " + str(prediction_value[model_name]))

            if confidence > 75:
                re_save = Check.objects.create(check="True")
                re_save.save()

                print("Confidence : " + str(confidence))
                print("same!" + " welcome " + str(name))
            elif confidence < 75:
                print("Confidence : " + str(confidence))
                print("Fail!")

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


##이미지 저장 및 불러오는 건 성공
class get_image(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        res = Respone.objects.all()
        re_len = len(res)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_filter.detectMultiScale(gray, 1.3)
        if faces is ():
            pass
        else:
            # dir = "faces/"
            print("is")

            for (x, y, w, h) in faces:
                Re = Respone.objects.create(respone="True")
                Re.save()
                face_image = image[y:y + h, x:x + h]
                image = cv2.flip(image, 0)
                face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                users = User.objects.all().order_by('-datetime')
                name = users[0]
                # print(name)  # name = User.objects.all()[0]

                # print(name)
                if re_len < 100:
                    cv2.imwrite("media/" + dir + str(name) + "/" + "user" + str(re_len) + '.JPG', face)  # 로컬저장

                    img = Photo.objects.create(title=str(name),
                                               images=dir + str(name) + "/" + "user" + str(
                                                   re_len) + '.JPG')  ##title 설정 고민해야함 #경로 절대 건들지 마삼 서버저장
                    img.save()

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
