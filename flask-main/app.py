import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template

from constants import *

# Disabling tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Defining cnn model
CNN_MODEL = load_model('models/local_model_10.h5')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Getting the image from user
        f = request.files['image']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'uploads', f.filename)
        print("Uploaded image: ", file_path)
        f.save(file_path)

        # Reshaping the image
        img = image.load_img(file_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Making predictions
        predictions = CNN_MODEL.predict(x)
        print("Predictions: ", predictions)

        # Generating batches of tensor image data with real-time data augmentation
        generator = ImageDataGenerator()
        train_ds = generator.flow_from_directory(DIR_TRAIN, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE_32)

        #
        classes = list(train_ds.class_indices.keys())
        print("Bird specie: ", classes[np.argmax(predictions)])

        #
        probability = round(np.max(CNN_MODEL.predict(x) * 100), 2)
        print("Probability: ", probability)

        # Printing the output message
        text = "La classe de l'oiseau sur l'image passée est : " + str(classes[np.argmax(predictions)]) \
            + " avec une probabilité de " + str(probability) + "%"

    return text


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
