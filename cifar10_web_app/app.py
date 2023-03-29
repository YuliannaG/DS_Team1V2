from imageio.v2 import imread
from flask import Flask, render_template, request
from skimage.transform import resize
import sys
import os

sys.path.append(os.path.abspath("./model"))

from load_model import *

graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/upload/', methods=['GET', 'POST'])
def predict():
    file = request.files['file']
    filename = 'output.png'
    file.save(os.path.join('static', 'uploads', filename))
    img = imread('static/uploads/output.png', pilmode='L')
    x = imread('static/uploads/output.png', pilmode='L')
    x = np.invert(x)
    x = resize(x, (32, 32, 3))
    print(x.shape)
    x = x.reshape(-1, 32, 32, 3)

    with graph.as_default():
        model = init()
        label_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        predict_img = model.predict(x)
        predict_class = label_names[np.argmax(predict_img)]
        response = str(f'Its - {predict_class}')
        return response


if __name__ == '__main__':
    app.run(debug=True, port=8000)
