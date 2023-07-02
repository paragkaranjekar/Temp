from flask import Flask, request, render_template
import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

model = load_model('project\hdr_model.h5', custom_objects={'Adam': Adam})
img_size = 28


def find_digit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 120
    gray[gray > threshold] = 255
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    resized = cv2.resize(thresh, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = np.array(newimg).reshape(-1, img_size, img_size, 1)
    predictions = model.predict(newimg)

    return str(np.argmax(predictions))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'code'


@app.route('/', methods=['POST', 'GET'])
def handle_upload():
    # Handle the data URL here
    data_url = request.form.get('data_url')
    answer = ""
    # Process the data URL as needed
    if data_url:
        image_data = data_url.split(',')[1]
        decoded_image_data = base64.b64decode(image_data)
        image_array = np.frombuffer(decoded_image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        answer = find_digit(image)

    return render_template('index.html', results=answer)


if __name__ == '__main__':
    app.run()
