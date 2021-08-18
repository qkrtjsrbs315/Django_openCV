import datetime as datetime
from django.db import models


# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=10, null=False, unique=False)
    datetime = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

    class Meta:
        ordering = ['datetime']


class Photo(models.Model):
    title = models.CharField(max_length=255)
    images = models.FileField()

    def __str__(self):
        return self.title


class Respone(models.Model):
    respone = models.CharField(max_length=20)

    def __str__(self):
        return self.respone


class Check(models.Model):
    check = models.CharField(max_length=20)

    def __str__(self):
        return self.check


class Login(models.Model):
    username = models.CharField(max_length=20)
    datetime = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

    class Meta:
        ordering = ['datetime']
