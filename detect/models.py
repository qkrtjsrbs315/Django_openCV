from django.db import models


# Create your models here.

class Post(models.Model):
    username = models.CharField(max_length=10, null=False, unique=False)

    def __str__(self):
        return self.username


class Photo(models.Model):
    title = models.CharField(max_length=255)
    images = models.ImageField(upload_to='images')

    def __str__(self):
        return self.title