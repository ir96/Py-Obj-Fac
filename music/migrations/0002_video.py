# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2018-02-09 15:25
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('music', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('videox', models.FileField(upload_to='')),
                ('videoname', models.CharField(max_length=250)),
            ],
        ),
    ]