"""
Definition of urls for DjangoWebProject1.
"""

from datetime import datetime
from django.urls import path, include, re_path
from django.contrib import admin
from . import views

from django.shortcuts import render
from .models import Diseases
from django.conf import settings





app_name = 'plant'
urlpatterns = [
    path('',views.IndexClass.as_view(),name="index"),
    path('diseases_list/',views.DiseasesList.as_view()),
    path('diseases_list/<int:pk>/',views.DiseasesDetail.as_view()),
    path('predict_list/',views.PredictList.as_view()),
    path('predict_list/<int:pk>/',views.PredictDetail.as_view()),
    path('ai/',views.TestBase.as_view()),
]



