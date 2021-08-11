import numpy as np
from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponse
import cv2
import threading
from PIL import Image
import glob

from django.core.files import temp as tempfile

from detect.detect import VideoCamera
from detect.models import Photo, Post

face_filter = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def register(request):
    if request.method == "GET":
        return render(request, 'register.html')

    elif request.method == "POST":
        username = request.POST.get('username')
        members = Post.objects.all()

        for user in members:
            if username == user.username:
                print("error!")
                return HttpResponse(
                    '<script>alert("이미 있는 유저입니다!");location.href="http://127.0.0.1:8000/register";</script>')

            else:
                new_user = Post.objects.create(username=username)
                new_user.save()
                request.session['username'] = username

                break

        return redirect('image')


def register_image(request):
    if request.method == "GET":
        return render(request, 'register_image.html')

    elif request.method == "POST":
        face_filter = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        username = request.session['username']
        images = request.FILES.getlist('images')

        print(images)
        # for image in images:
        #     file_temp = tempfile.NamedTemporaryFile()
        #     file_temp.write(image.read())
        #     image_temp_name = file_temp.name

        for image in images:
            # print(image.url)
            # gray = cv2.cvtColor(image.url, cv2.COLOR_BGR2RGB)
            # img = cv2.imread(image.url)
            # print(img)
            # print(gray)
            # face = face_filter.detectMultiScale(gray, 1.3, 5)

            new_image = Photo()
            new_image.title = username
            new_image.images = image
            new_image.save()

        return redirect('login')


def find(img):
    face_filter = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_filter.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        re = img[y:y + h, x:x + w]

    face = cv2.resize(re, (200, 200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


def train():
    face_filter = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def login(request):

    return render(request, 'login.html')


def home(request):
    return render(request, "home.html")


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    cam = VideoCamera()
    return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
