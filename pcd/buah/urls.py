from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import include
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required

from . import views

app_name = 'buah'
urlpatterns = [
    url(r'^$', views.home.as_view(), name='home'),
    url(r'^hasil/(?P<oy>[a-zA-Z0-9_]+)/$', views.hasil.as_view(), name='hasil'),
    url(r'^hasils/proses/$', views.upload_pic, name='masukin'),
    #url(r'^updatedata/$', views.crud.updatedata, name='updatedata'),
    #url(r'^([0-9]+)/editdata/$', views.crud.editdata, name='editdata'),
    #url(r'^([0-9]+)/deletedata/$', views.crud.deletedata, name='deletedata'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

print 'urls.py MEDIA_URL: %s' % (settings.MEDIA_URL)
print 'urls.py MEDIA_ROOT: %s' % (settings.MEDIA_ROOT)
