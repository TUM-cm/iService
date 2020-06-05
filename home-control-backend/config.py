import os

SECRET_KEY = os.urandom(24) # uuid.uuid4().hex
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite')
SQLALCHEMY_TRACK_MODIFICATIONS = False
