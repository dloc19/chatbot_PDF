from django.urls import path
from . import views


urlpatterns = [
    path('', views.chatGoD, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('account/', views.account, name='account'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload, name='upload'),
    path('selected/', views.select_files, name='select_file'),
]
