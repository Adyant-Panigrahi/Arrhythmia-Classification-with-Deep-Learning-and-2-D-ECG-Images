from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
model = load_model('ECG.h5')

# Use static/uploads for serving images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class_labels = [
    'Left Bundle Branch Block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contractions',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    img_filename = None
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            filename = secure_filename(img_file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            img_file.save(img_path)
            img = image.load_img(img_path, target_size=(128,128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds, axis=1)[0]
            prediction = class_labels[predicted_class]
            img_filename = filename
    return render_template('index6.html', prediction=prediction, img_filename=img_filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
