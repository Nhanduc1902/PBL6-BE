from django.contrib import admin

# Register your models here.
from .models import Diseases, Predict
admin.site.register(Diseases)
admin.site.register(Predict)