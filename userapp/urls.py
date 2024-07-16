from django.urls import path # type: ignore
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('selects/',views.selects, name="selects"),
    path('answer',views.answer, name="answer"),
    path('item/<str:item_id>/', views.get_item_description, name='get_item_description'),

    path('calculate/', views.calculate_and_plot, name='calculate_and_plot'),
    path('export_csv/<str:filename>/', views.export_csv, name='export_csv'),


    path('api/result-chart/', views.functiondata, name='functionData'),
]
