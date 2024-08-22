import requests
import numpy as np
import cv2
import json
import json_numpy
import time
import sys
from matplotlib import pyplot as plt


def main():

    image_path = "0/color_0_png/10.73110556602478_color_0.png"
    image = cv2.imread(image_path)

    image = cv2.resize(image, (320, 256))
    json_numpy.patch()
    payload = {"image": image, "instruction": "move the robot arm to the right", "camera_type": "side_view"}

    # Send the POST request
    obs = requests.post(
        "http://0.0.0.0:8000/action",
        json=payload,
    ).json()
    print(type(obs))
    action = obs
    print("Action:", action)


if __name__ == "__main__":
    main()
