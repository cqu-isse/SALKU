import tensorflow as tf
from SALKU_reimplementation.SALKU_model import SALKU_model

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = SALKU_model()
    model.SALKU_train()
    model.SALKU_test()








