# Import
#
import cv2
import os
import numpy as np
import tensorflow as tf
import time
def tf_record_reader(file_name):
    record_iterator = tf.python_io.tf_record_iterator(path=file_name)
    dataset = []
    target = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        frame = int(example.features.feature['frame'].int64_list.value[0])
        label = example.features.feature['label'].bytes_list.value[0]
        video_str = example.features.feature['video'].bytes_list.value[0]

        video_1d = np.fromstring(video_str, dtype=np.uint8)
        video = video_1d.reshape((frame,height,width))
        dataset.extend(video)
        target.extend(label)
    return dataset, target

start = time.time()
train_dataset,train_label = tf_record_reader('KTH_Train_Data.tfrecords')
test_dataset,test_label = tf_record_reader('KTH_Train_Data.tfrecords')
end = time.time()
print('Data Loaded From TFRecord in:', end-start)
print ('Dataset Size:', len(train_dataset)+len(test_dataset))

