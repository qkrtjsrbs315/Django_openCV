from django.conf.urls import url, include
from django.urls import path
from django.conf.urls.static import static

from webdetect import settings
from . import views

urlpatterns = [
    path('', views.login, name="index"),
    path('register/', views.register, name="register"),
    #path('register_image/',views.register_image, name="register_image"),
    path('detectme/', views.detectme, name="detectme"), ##detecting user
    path('login/',views.index,name="login"),
    path('detect_image/', views.detect_image, name="detect_image"), ##save the image in webcam
    #path('select/', views.select, name="select"),
    path('save_image/', views.save_image, name="save_image"),
    path('user/', views.user, name="user"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)