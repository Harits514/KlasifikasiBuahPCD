# -*- coding: utf-8 -*-
# Generated by Django 1.11.13 on 2018-05-21 04:58
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('buah', '0006_auto_20180521_1155'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fruit',
            name='document',
            field=models.FileField(blank=True, upload_to='documents/'),
        ),
    ]