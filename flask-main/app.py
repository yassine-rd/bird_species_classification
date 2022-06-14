import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DIR_BASE = 'test_db'
DIR_TRAIN = os.path.join(DIR_BASE, 'train')

my_model = load_model('models/local_model_10.h5')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("Uploaded image: ", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        predictions = my_model.predict(x)
        print("Predictions: ", predictions)

        generator = ImageDataGenerator()
        train_ds = generator.flow_from_directory(DIR_TRAIN, target_size=(224, 224), batch_size=32)

        classes = list(train_ds.class_indices.keys())
        print("Bird specie: ", classes[np.argmax(predictions)])

        probability = round(np.max(my_model.predict(x) * 100), 2)
        print("Probability: ", probability)

        text = "La classe de l'oiseau sur l'image passée est : " + str(classes[np.argmax(predictions)]) + " avec une probabilité de " + str(probability)

    return text


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
