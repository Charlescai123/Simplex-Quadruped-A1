import tensorflow as tf
import os
import numpy as np
import time


converter = tf.lite.TFLiteConverter.from_saved_model(
    "tf_models/actor")  # path to the SavedModel directory

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

# tflite_model = converter.convert()

#
save_path = 'tf_models/tf_lite/'
# os.mkdir(save_path)
# # Save the model.
# with open(save_path+'model.tflite', 'wb') as f:
#     f.write(tflite_model)



TFLITE_FILE_PATH = save_path+'model.tflite'

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

running_time = []
running_episode = 1
all_time = []

for i in range(running_episode):

    for step in range(100):

        time_start = time.time()
        observations_input = np.random.random((1, 12)).astype(np.float16)

        interpreter.set_tensor(input_details[0]['index'], observations_input)
        interpreter.invoke()

        action = interpreter.get_tensor(output_details[0]['index'])
        acton = np.array(action).squeeze()
        all_time.append(time.time() - time_start)



print(np.mean(all_time))


