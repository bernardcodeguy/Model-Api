import os

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms import transforms

from pre_processing import pre_process_image
IMG_PATH = os.path.join(os.getcwd(), 'image.png')
DEVICE = torch.device("cpu")


transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




def predict_image(image):
    # pre-process the image
    pre_processed_image = pre_process_image(image)

    # get image ready for prediction
    input_image = transform_pipeline(pre_processed_image)
    pre_processed_image.close()
    os.remove(IMG_PATH)
    input_image = input_image.unsqueeze(0)
    input_image = input_image.to(DEVICE)

    # get the pre-trained model
    model = torch.load('model.pth', map_location=DEVICE)
    model.eval()

    # get prediction
    prediction = model(input_image)
    prediction = functional.softmax(prediction.float(), dim=1)
    value, index = torch.max(prediction, 1)

    value = int(value[0] * 100)
    if index[0] == 0:
        prediction = {"classification": "Benign", "percentage": value}
    else:
        prediction = {"classification": "Malignant", "percentage": value}

    return prediction
