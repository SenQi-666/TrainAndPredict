import keras
import cv2
from keras.models import load_model
import numpy as np
import os

average = 118.6506
LABELS = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']


def Loadlmage(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    img -= average
    img /= 255.
    return np.array(img[:, :, :3])


if __name__ == "__main__":
    model = load_model("./vggmodel_20-0.30-0.89.hdf5")
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adadelta(),
                 metrics=["accuracy"])
    path_base = r"kesxyt"
    subbdir = os.listdir(path_base)
    subbdir.sort()
    i = os.listdir(path_base)
    img = Loadlmage(os.path.join(path_base, i[0]))
    res = np.argmax(model.predict(np.array([img])))
    print(LABELS[res])
