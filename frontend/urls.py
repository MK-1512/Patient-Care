
from django.urls import path
from . import views 

urlpatterns = [
        path('', views.index, name='index'),
        path('activity/', views.activity_report, name='activity_report'),
        path('run-detection/', views.run_detection, name='run_detection'),
        path('how-it-works/', views.how_it_works, name='how_it_works'),
        path('contact/', views.contact, name='contact'),
        path('team/', views.team, name='team'),
        path('upload-js/', views.image_upload_js_view, name='upload_image_js'),

        path('recordings/', views.view_recordings, name='view_recordings'),
    ]

