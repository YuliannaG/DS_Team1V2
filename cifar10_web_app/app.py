from flask import Flask, render_template, request
from imageio.v2 import imread
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
    x = imread('static/uploads/output.png', pilmode='L')
    x = np.invert(x)
    x = resize(x, (32, 32, 3))
    print(x.shape)
    x = x.reshape(1, 32, 32, 3)

    # image.reshape(3, 32, 32).transpose(1, 2, 0)

    with graph.as_default():
        model = init()
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))

        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    app.run(debug=True, port=8000)
