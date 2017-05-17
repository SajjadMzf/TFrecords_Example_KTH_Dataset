# Import
#
import cv2
import os
import numpy as np
import zipfile
import tensorflow as tf
import time
target_dir = "./Dataset/"
fps = 25
max_frames = 5*fps
frame_height = 120
frame_width = 160
train_all_ratio =0.7
# Load Dataset
#
print('Loading Dataset...')
def extract_dataset(dir, dataset_file):
    dataset_folder = os.path.splitext(os.path.splitext(dataset_file)[0])[0]
    dataset_dir = os.path.join(dir,dataset_file)
    if os.path.exists(os.path.join(dir, dataset_folder)):
        print("Already extracted!")
        return
    if not os.path.exists(dataset_dir):
        print(dataset_file, "not found!")
        return
    _zip = zipfile.ZipFile(dataset_dir, 'r')
    _zip.extractall(os.path.join(dir, dataset_folder))
    _zip.close()
    os.remove(dataset_dir)
    print(dataset_file, "extracted!")
extract_dataset(target_dir, 'boxing.zip')
extract_dataset(target_dir, 'handclapping.zip')
extract_dataset(target_dir, 'handwaving.zip')
extract_dataset(target_dir, 'jogging.zip')
extract_dataset(target_dir, 'running.zip')
extract_dataset(target_dir, 'walking.zip')


def load_vid(path, filename, max_frame):
    cap = cv2.VideoCapture(os.path.join(path,filename))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = 0
    buf = np.empty((max_frame, frameHeight, frameWidth), np.dtype('uint8'))
    if frameCount <max_frame:
        print('Invalid video duration:', filename, '-Skipping')
        return buf,fc
    ret = True
    while (fc < max_frame and ret):
        ret, temp = cap.read()
        buf[fc] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        fc += 1
    cap.release()
    return buf

#@profile
def load_class(dir,class_folder):
    class_dir = os.path.join(dir, class_folder)
    if not os.path.exists(class_dir):
        print(class_folder, "not found!")
        return
    data_files = os.listdir(class_dir)
    class_dataset = np.ndarray(shape=(len(data_files)*max_frames,frame_height,frame_width), dtype='uint8')
    data_count = 0
    for file_name in data_files:
        try:
            data = load_vid(class_dir, file_name, max_frames)
            if data.shape !=(max_frames,frame_height,frame_width):
                print('Unexpected Data size:', file_name, '-skipping')
            class_dataset[data_count*max_frames:(data_count+1)*max_frames, :,:] = data
            data_count = data_count + 1
            del data
        except IOError as e:
            print(e, '-skipping')
    print(class_dir, 'Loaded','Class Size:', data_count)
    return class_dataset, data_count

#@profile
def load_dataset(dir, dataset_folder):
    dataset_dir = os.path.join(dir, dataset_folder)
    if not os.path.exists(dataset_dir):
        print(dataset_folder, "not found!")
    class_files = sorted(os.listdir(dataset_dir))
    class_num = len(class_files)
    dataset = []
    targets = []
    dataset_size = 0
    for idx in range(class_num):
        class_data, class_size = load_class(dataset_dir, class_files[idx])
        dataset.extend(class_data[:class_size*max_frames,:,:])
        targets.extend(idx*np.ones(shape = class_size, dtype='uint8'))
        dataset_size = dataset_size + class_size

    return dataset, targets, class_num, dataset_size

start = time.time()
dataset, labels, class_num, dataset_size = load_dataset(target_dir,'')
fullIdx = range(dataset_size)
fullIdx = np.array(fullIdx)
np.random.shuffle(fullIdx)
trIdx = fullIdx[:int(train_all_ratio*len(fullIdx))]
teIdx = fullIdx[int(train_all_ratio*len(fullIdx)):]
end = time.time()
print('Data loaded from video files in:', end-start)
print('Dataset Size:', len(dataset))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_record_writer(file_name,all_data,all_labels,Idx):
    writer = tf.python_io.TFRecordWriter(file_name)
    for example_idx in Idx:
        features = all_data[example_idx*max_frames:(example_idx+1)*max_frames]
        features = np.array(features)
        label = all_labels[example_idx]
        seq_features = features.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': _bytes_feature(label),
                    'height': _int64_feature(frame_height),
                    'width':_int64_feature(frame_width),
                    'frame':_int64_feature(max_frames),
                    'video':_bytes_feature(seq_features)
                }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    print (file_name, 'Writed!')

tf_record_writer('KTH_Train_Data.tfrecords', dataset, labels, trIdx)
tf_record_writer('KTH_Test_Data.tfrecords', dataset, labels, teIdx)
