import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

key = ".firebase-venv/api_key.json"

if __name__ == "__main__":
    cred = credentials.Certificate(key)
    firebase_admin.initialize_app(cred)

