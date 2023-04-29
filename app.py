from flask import Flask, render_template, request, jsonify, session
import cv2
import time
import numpy as np
from tensorflow import keras

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_message")
def get_message():
    model = keras.models.load_model('model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    last_time = time.time()
    last_emotion = None
    name = request.args.get('name', '')
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        listening = False

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 48, 48, 1)

            emotion = model.predict(face_img)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            num_categories = model.output_shape[1]
            categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

            if np.argmax(emotion) < len(categories):
                emotion_label = categories[np.argmax(emotion)]
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if emotion_label == "Neutral":
                    listening = True

        if time.time() - last_time > 15:
            if listening:
                message = f"{name}: listening"
            else:
                message = f"{name}: not listening"

            if message != last_emotion:
                last_emotion = message
                last_time = time.time()
                session['status'] = message

    cap.release()
    cv2.destroyAllWindows()

@app.route("/get_status")
def get_status():
    return jsonify(session.get('status', ''))

if __name__ == "__main__":
    app.run(debug=True)
