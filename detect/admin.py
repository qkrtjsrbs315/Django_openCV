from django.contrib import admin

# Register your models here.
from detect.models import User, Photo, Respone, Check, Login

admin.site.register(User)
admin.site.register(Photo)
admin.site.register(Respone)
admin.site.register(Check)
admin.site.register(Login)