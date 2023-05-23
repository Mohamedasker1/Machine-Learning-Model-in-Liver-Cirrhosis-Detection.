from flask import Flask, request, render_template
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the saved model
saved_model = load_model("liver_model.h5")


@app.route('/')
def home():
    return render_template('index.html')


# Route to handle file upload from frontend
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    # Load image and preprocess it
    img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #img_file = request.files['file']
    #img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)


    img = cv2.resize(img, (256, 256))
    img = img / 255.0


    # Make a prediction using the loaded model
    prediction = saved_model.predict(np.array([img]))

    # Determine the predicted class
    output = ""
    if prediction[0] > 0.5:
        output = "Cirrhotic Liver"
    else:
        output = "Normal Liver"
    print(output)
    # Display the result on HTML page
    return render_template('result.html', prediction=output)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7000)
