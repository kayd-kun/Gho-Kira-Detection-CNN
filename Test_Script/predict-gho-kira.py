import pickle
import pathlib 
import numpy as np
import tensorflow as tf

# Import modle
model = pickle.load(open('deploy-model-76-no-overfitting.pkl', 'rb'))

# Define Parameters
class_names = ['gho', 'kira']
img_height = 240
img_width = 240

# Predict Image function
def predict_image(pathToImg):
    new_data_path = pathlib.Path(pathToImg)

    img = tf.keras.utils.load_img(
        new_data_path, target_size = (img_width, img_height)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Creating a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# Code to test the image
predict_image('./test-kira-1.jpg')
predict_image('./test-kira-2.jpg')
predict_image('./test-gho-1.jpg')
predict_image('./test-gho-2.jpg')

