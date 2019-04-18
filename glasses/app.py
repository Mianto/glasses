import io
import json
import base64
from flask import Flask, request, redirect, url_for, render_template, jsonify
import torch
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
from werkzeug import secure_filename

app = Flask(__name__)
use_gpu = False
classes = ['glasses', 'no-glasses']
app.config['UPLOAD_FOLDER'] = r"C:\Users\Siddhant\Documents\Glasses\glasses\files"
app.config["SECRET_KEY"] = "secret key"

def load_model(path):

    model = torch.load(path)
    model.eval()
    if use_gpu:
        model.cuda()
    return model

def prepare_image(image):
    if image.mode != ' RGB':
        image = image.convert('RGB')
    
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])(image)

    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if request.method == "POST":
        image = base64.decodestring(request.form['image'].encode('utf-8'))
        # Read file in PIL format
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image)
        model = load_model(r"C:\Users\Siddhant\Documents\Glasses\glasses\model.pth")
        out = model(image)
        _, predicted = torch.max(out, 1)

        data['prediction'] = classes[predicted]
        data['success'] = True
        print(data)
        
    return jsonify(data)

if __name__=='__main__':
    print("Loading pytorch and flask")
    print("Please wait ...")
    app.run(debug=True)


