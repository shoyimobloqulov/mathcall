from django.urls import path # type: ignore
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('selects/',views.selects, name="selects"),
    path('answer',views.answer, name="answer"),
    path('item/<str:item_id>/', views.get_item_description, name='get_item_description'),
    path('export_csv/<str:filename>/', views.export_csv, name='export_csv'),
    path('about', views.about, name='about'),
    path('calculate/', views.calculate, name='calculate'),
    path('result/',views.result,name='result')
]
