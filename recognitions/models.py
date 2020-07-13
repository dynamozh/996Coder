from django.db import models

# Create your models here.


class PREIMG(models.Model):
    img = models.FileField(upload_to='media')

    def __str__(self):
        return str(self.id)


class IMG(models.Model):
    img = models.FileField(upload_to='media')

    def __str__(self):
        return str(self.id)