from PIL import Image
from flask import request, jsonify, Flask
import predictor

app = Flask(__name__)


@app.route('/skin-cancer-diagnosis', methods=['POST'])

def diagnose_skin_lesion():

    if not request.files:
        return jsonify(), 400

    file = request.files['image']
    image = Image.open(file)


    prediction = predictor.predict_image(image)
    return jsonify(prediction), 201



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)