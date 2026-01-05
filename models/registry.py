from models.image.alexnet import load_model as load_alexnet
from models.image.densenet import load_model as load_densenet
from models.audio.mobilenet import load_model as load_mobilenet
from models.audio.vgg16 import load_model as load_vgg16

MODEL_REGISTRY = {
    "image": {
        "alexnet": {
            "name": "AlexNet",
            "loader": load_alexnet,
            "input_size": (224, 224),
            "labels": ["fake", "real"],
        },
        "densenet": {
            "name": "DenseNet",
            "loader": load_densenet,
            "input_size": (224, 224),
            "labels": ["fake", "real"],
        },
    },
    "audio": {
        "mobilenet": {
            "name": "MobileNet",
            "loader": load_mobilenet,
            "input_size": (224, 224),
            "labels": ["fake", "real"],
        },
        "vgg16": {
            "name": "VGG16",
            "loader": load_vgg16,
            "input_size": (224, 224),
            "labels": ["fake", "real"],
        },
    },
}
