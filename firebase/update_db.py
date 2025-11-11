import firebase_credential
from firebase_admin import firestore
import socket

firebase_credential.verify()

db = firestore.client()
collection = db.collection("training_data")
print(socket.gethostname())


