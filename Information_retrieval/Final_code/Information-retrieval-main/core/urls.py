from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("search/", views.search, name="search"),
    path("classify/", views.classify, name="classify"),
    path("model-selection/", views.model_selection, name="model_selection"),
]
