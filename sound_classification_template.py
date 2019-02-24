import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.metrics as metrics
import math
from matplotlib.pyplot import specgram

# librosa imports
import librosa
import librosa.display  # need to import specifically
import librosa.core  # need to import specifically


# NOTE: must be run within same folder as sound file folders


############################### USEFUL FUNCTIONS ###############################

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


################################# VISUALIZATION ################################

def plot_waves_orig(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_waves(sound_names,raw_sounds):
    i = 1
    # plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        # plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        # plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        # plt.subplot(10,1,i)
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f))**2, ref=np.max) # core.amplitude_to_db replaced logamplitude
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

sound_file_paths = ["air_conditioner/101729.wav"] # ADJUST ACCORDINGLY

sound_names = ["air conditioner"] # ADJUST ACCORDINGLY

raw_sounds = load_sound_files(sound_file_paths)

# TEMP: UNCOMMENT TO DISPLAY
# plot_waves(sound_names,raw_sounds)
# plot_specgram(sound_names,raw_sounds)
# plot_log_power_specgram(sound_names,raw_sounds)


############################### DATA EXTRACTION ################################

def extract_feature(file_name):
    # EXPLANATION: define sounds by their features to represent their relative similarities
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files_old(parent_dir,sub_dirs,file_ext="*.wav"):
    # instantiate labels to 0, dtype = int
    features, labels = np.empty((0,193)), np.empty(0, int)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(fn)
            try:
              mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue

            # EXPLANATION: we are going through a sub_dir (containing sample sounds from one specific class)
                # thus, all sounds in this sub_dir have the same numerical label (which is arbitrary)
                # each row r in vstack, contains hstack with features corresponding to some sound
                # labels keeps track of label of row r in vstack (so we know which class features correspond to)

            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)

    print("features:")
    print(np.array(features))
    print("labels:")
    print(np.array(labels))

    return np.array(features), np.array(labels)

def parse_audio_files(parent_dir,sub_dir_labels,file_ext="*.wav"):
    # instantiate labels to 0, dtype = int
    features, labels = np.empty((0,193)), np.empty(0, int)
    for sub_dir, label in sub_dir_labels:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(fn)
            try:
              mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue

            # EXPLANATION: we are going through a sub_dir (containing sample sounds from one specific class)
                # thus, all sounds in this sub_dir have the same numerical label (which is arbitrary)
                # each row r in vstack, contains hstack with features corresponding to some sound
                # labels keeps track of label of row r in vstack (so we know which class features correspond to)

            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)

    return np.array(features), np.array(labels)

# EXPLANATION: creates one hot encode bool (1/0) 2D matrix - this is fed into the neural network
# one hot encode converts our numerical class labels into a boolean table
#    EX: 2 classes total - class 0 has two sound samples, class 1 has one sound sample
#        labels = [0 0 1]
#        resulting one hot encode = [[1 0], [1 0], [0 1]]
def one_hot_encode(labels, num_training_classes):
    n_labels = len(labels)
    one_hot_encode = np.zeros((n_labels,num_training_classes), int)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = 'sample_sound_sets'

# tr_sub_dirs = ["air_conditioner_mini", "car_horn_mini", "children_playing_mini"] # training set
# ts_sub_dirs = ["air_conditioner_test", "car_horn_test", "children_playing_test"] # test set
# tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
# ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

# array of directories - each dir is specific to a class -> [directory for class, numeric label for class]
tr_sub_dirs_labels = [["air_conditioner_mini",0], ["car_horn_mini",1], ["children_playing_mini",2]] # training set
ts_sub_dirs_labels = [["air_conditioner_test",0], ["car_horn_test",1], ["children_playing_test",2]] # test set
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs_labels)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs_labels)

num_training_classes = len(np.unique(tr_labels))
tr_labels = one_hot_encode(tr_labels, num_training_classes)
ts_labels = one_hot_encode(ts_labels, num_training_classes)


################################ NEURAL NETWORK ################################

# neural net configuration
training_epochs = 50
n_dim = tr_features.shape[1] # shape[1] - # of cols in tr_features (or training set)
print("n_dim: {}".format(n_dim))
print("unique: {}".format(np.unique(tr_labels)))
n_classes = 3
# n_classes = len(np.unique(tr_labels)) # number of classes in training set (how many categories are we classifying)
print("num classes: {}".format(n_classes))
n_hidden_units_one = 280 # ADJUST ACCORDINGLY
n_hidden_units_two = 300 # ADJUST ACCORDINGLY
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01 # ADJUST ACCORDINGLY (how quickly the models learns)



# EXPLANATION: create placeholders which tensorflow will fill with data
X = tf.placeholder(tf.float32,[None,n_dim]) # type = float32, None = any num rows, n_dim (or # features) cols of data
Y = tf.placeholder(tf.float32,[None,n_classes]) # type = float32, None = any num rows, n_classes cols of data



# EXPLANATION: create variables - 2 layers are typically sufficient for little data

# layer 1
W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

# layer 2
W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two],mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.initialize_all_variables()



# EXPLANATION: use gradient descent to cluster similar sounds based on their features
# gradient descent moves in a direction that minimizes the cost as defined by cost_function
# cost_function:
#       uses cross entropy where Y is binary indicator (whether or not prediction is correct)
#       and y_ represents the predicted probability that that item is of correct class
#           lower predicted prob -> smaller log(y_) -> increased cost (because negated)
#           higher predicted prob -> larger log(y_) -> increased cost (because negated)
cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# prediction model
cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    # EXPLANATION: training the gradient descent optimizer for training_epoch iterations
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        # track the cost of each training iteration to plot later
        cost_history = np.append(cost_history,cost)

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))

    print("Predictions: {}".format(y_pred))
    print("Actual: {}".format(y_true))

    # EXPLANATION: determine how well the trained optimizer works on the test set
    #       by comparing predicted labels of sounds in training set to actual labels
    accuracy_output = sess.run(accuracy, feed_dict={X: ts_features,Y: ts_labels})
    print("Test accuracy: {0:.3f}".format(accuracy_output))

# EXPLANATION: can visualize how well the neural network predicts for each class
#       - most common mistake for each class (most similar sound that it's confusing)
#       - accuracy for that class's test set
#       - (can update function to include more)
def visualize_results(y_pred, y_true):
    results = []

    for c in set(y_true):
        print(c)
        total = 0.0
        correct = 0

        confusion = {}

        for i in range(len(y_true)):
            if y_true[i] == c:
                pred = y_pred[i]
                if  pred == c:
                    correct += 1
                else:
                    if pred in confusion.keys():
                        confusion[pred] += 1
                    else:
                        confusion[pred] = 1
                total += 1

        most_confused_value = -math.inf
        most_confused_key = None
        for key, value in confusion.items():
            if value > most_confused_value:
                most_confused_value = value
                most_confused_key = key

        accuracy = correct/total

        bold = '\033[1m'
        reset = '\033[0m'
        result = "{0}Class {1}:{2} \n\t accuracy: {3:2f} \n\t most confused with: {4}".format(bold, c, reset, accuracy, most_confused_key)
        results.append(result)

    for r in results:
        print(r)

    return

visualize_results(y_pred, y_true)

# plotting results
fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
max_cost = np.max(cost_history)

if max_cost is None or max_cost == math.inf:
    max_cost = 10
plt.axis([0,training_epochs,0,max_cost])
plt.show()

# recall, update_op = metrics.recall(y_true, y_pred)
# print("F-Score: {0:3f}".format(recall))
