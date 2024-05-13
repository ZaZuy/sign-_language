# import  pandas as pd
# import  numpy as np
# import tflite_runtime.interpreter as tflite
# ROWS_PER_FRAME = 543  # number of landmarks per frame
#
# def load_relevant_data_subset(pq_path):
#     data_columns = ['x', 'y', 'z']
#     data = pd.read_parquet(pq_path, columns=data_columns)
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)
#
#
# interpreter = tflite.Interpreter("model.tflite")
# found_signatures = list(interpreter.get_signature_list().keys())
# prediction_fn = interpreter.get_signature_runner("serving_default")
#
# # output = prediction_fn(inputs=demo_raw_data)
# # sign = output['outputs'].argmax()
#
# train = pd.read_csv('train.csv')
# pq_file = 'output_0_27.parquet'
# train['sign_ord'] = train['sign'].astype('category').cat.codes
# # Dictionaries to translate sign <-> ordinal encoded sign
# SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
# ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
# xyz = load_relevant_data_subset(pq_file)
# predict = prediction_fn(inputs=xyz)
# print(predict['outputs'])
# sign = predict['outputs'].argmax()
# predicted_label = ORD2SIGN[sign]
# predicted_value = predict['outputs'][sign]
# accuracy = predicted_value * 100 / predict['outputs'].sum()
#
# print("Predicted Action:", predicted_label)
# print("Accuracy:", accuracy, "%")
# # print(train['sign'].unique)
#
import glob
import os
parquet_files = glob.glob("*.parquet")

num_parquet_files = len(parquet_files)
from collections import Counter
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite

ROWS_PER_FRAME = 543  # số lượng landmarks mỗi frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

interpreter = tflite.Interpreter("model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

n = num_parquet_files
my_predict = []
for i in range(n-1):
    pq_file = f'output_{i}.parquet'
    xyz = load_relevant_data_subset(pq_file)
    predict = prediction_fn(inputs=xyz)
    sign = predict['outputs'].argmax()
    predicted_label = ORD2SIGN[sign]
    predicted_value = predict['outputs'][sign]
    accuracy = predicted_value * 100 / predict['outputs'].sum()

    my_predict.append(predicted_label)
    print(f"Predicted Action for {pq_file}: {predicted_label}, Accuracy: {accuracy:.2f}%")
    os.remove(pq_file)
label_counts = Counter(my_predict)


for label, count in label_counts.items():
    if count < 2:
        my_predict = [item for item in my_predict if item != label]
unique_chars = list(set(my_predict))
while "TV" in unique_chars:
    unique_chars.remove("TV")
print("Các nhãn sau khi loại bỏ:", unique_chars)


