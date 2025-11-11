import firebase_credential
from firebase_admin import firestore

firebase_credential.verify_credentials()

db = firestore.client()
doc_ref = db.collection("training_data").document("in_training")
doc = doc_ref.get()
print(doc.to_dict())
