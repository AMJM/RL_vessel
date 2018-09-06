import numpy as np
from tensorflow.keras.models import load_model


class Controller:
    def __init__(self, path):
        self.model = load_model(path)

    def select_action(self, input_vec):
        input_vec = np.expand_dims(np.expand_dims(input_vec, 0), 0)
        return np.argmax(self.model.predict(input_vec))


if __name__ == '__main__':
    model = Controller('model/model.h5')
    print(model.select_action([10, 15, 20]))