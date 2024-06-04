#!/bin/sh

gunicorn --bind :$PORT --workers 2 'app:main'
