from django.db import models
from datetime import datetime  


# Create your models here.
class Diseases(models.Model):
    NameDisease_VN = models.CharField(max_length=100)
    NameDisease_ENG = models.CharField(max_length=100)
    Symptom = models.CharField(max_length=2000)
    Pathogens = models.CharField(max_length=2000)
    Treatment = models.CharField(max_length=2000)
    def __str__(self):
        return self.NameDisease_VN
    def to_json(self):
        return {
            'id':self.id,
            'NameDisease_VN': self.NameDisease_VN,
            'NameDisease_ENG':self.NameDisease_ENG,
            'Symptom':self.Symptom,
            'Pathogens':self.Pathogens,
            'Treatment':self.Treatment
        }

class Predict(models.Model):
    Image = models.URLField(max_length=1000)
    #DateTime = models.DateTimeField(auto_now_add=True)
    DateTime = models.DateTimeField(default=datetime.now, blank=True)
    PredictResult = models.ForeignKey(Diseases,on_delete=models.CASCADE)
    Confident = models.FloatField()

    def to_json(self):
        return {
            'id':self.id,
            'DateTime': self.DateTime,
            'Image':self.Image,
            'Confident':self.Confident,
            #'PredictResult': Diseases.objects.filter(pk=int(self.PredictResult)).first().to_json()
            'PredictResult': self.PredictResult.to_json()
        }
    
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')