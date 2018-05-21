# -*- coding: utf-8 -*-	
from __future__ import unicode_literals
import datetime
from django.db import models
from django.utils import timezone

class fruit(models.Model):
	Nama_Buah = models.CharField(blank=True, max_length=50)
	Tipe = models.IntegerField(null=True, blank=True)
	document = models.FileField(blank=True)
	
	class Meta:
		db_table = u'buah_fruit'

	def __str__(self):
		return self.Nama_Buah
