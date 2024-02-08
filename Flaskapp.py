from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os

app = Flask(__name__)

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ela_filename = filename.split('.')[0] + '.ela.png'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    ela_im.save(ela_filename)

    os.remove(resaved_filename)
    return ela_filename

# Load the trained model
loaded_model = load_model('fakevsreal_model_now.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)
            ela_img_path = convert_to_ela_image(image_path, 90)
            ela_img = Image.open(ela_img_path).resize((128, 128))
            X_pred = np.array(ela_img) / 255.0
            X_pred = np.expand_dims(X_pred, axis=0)

            # Predict
            prediction = loaded_model.predict(X_pred)
            predicted_class = np.argmax(prediction)
            realness_percentage = round(prediction[0][0] * 100,4)
            fakeness_percentage = round(prediction[0][1] * 100,4)

            result = "Real" if predicted_class == 0 else "Fake"
            return render_template('index.html', result=result, image_path=image_path,
                                   realness_percentage=realness_percentage, fakeness_percentage=fakeness_percentage)

    return render_template('index.html', result=None, image_path=None)
    
@app.route('/display_image/<filename>')
def display_image(filename):
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

