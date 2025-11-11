import firebase_admin
from firebase_admin import credentials

KEY = ".firebase-venv/api_key.json"

def verify_credentials(key=KEY):
    cred = credentials.Certificate(key)
    firebase_admin.initialize_app(cred)



