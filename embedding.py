from tensorflow.keras.models import load_model

class emb:
    def __init__(self):
        self.model=load_model('facenet_keras.h5',compile=False)
    def calculate(self,img):
        return self.model.predict(img)[0]
