from django.contrib import admin

# Register your models here.
from detect.models import Post, Photo

admin.site.register(Post)
admin.site.register(Photo)
