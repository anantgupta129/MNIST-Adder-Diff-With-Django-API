from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name='home'),
    path("try", views.predict, name='predict'),
    path("about", views.about, name="about")
]

