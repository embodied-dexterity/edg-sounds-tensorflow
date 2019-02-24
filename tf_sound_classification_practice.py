import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from matplotlib.pyplot import specgram

# librosa imports
import librosa
import librosa.display  # need to import specifically
import librosa.core  # need to import specifically


# NOTE: must be run within same folder as UrbanSound/data

############################### USEFUL FUNCTIONS ###############################

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


################################# VISUAL STUFF #################################

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

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
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

            # hstack - horizontal stack, vstack - vertical stack
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

# EXPLANATION: creates one hot encode bool (1/0) 2D matrix - this is fed into the neural network
# one hot encode converts our numerical class labels into a boolean table
#    EX: 2 classes total - class 0 has two sound samples, class 1 has one sound sample
#        labels = [0 0 1]
#        resulting one hot encode = [[1 0], [1 0], [0 1]]
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels)) # number of unique labels
    one_hot_encode = np.zeros((n_labels,n_unique_labels), int)
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = '.'

# tr_sub_dirs = ["temp3", "temp4"] # learning set (ADJUST ACCORDINGLY)
# ts_sub_dirs = ["temp3"] # test set (ADJUST ACCORDINGLY)

tr_sub_dirs = ["air_conditioner_mini", "car_horn_mini"]#, "children_playing"] # learning set
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features = tr_features # TEMP
ts_labels = tr_labels # TEMP
# ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

print("tr_labels:")
print(tr_labels)
print("ts_labels:")
print(ts_labels)


################################ NEURAL NETWORK ################################

# neural net configuration
training_epochs = 50
n_dim = tr_features.shape[1] # shape[1] - # of cols in tr_features (or training set)
print("n_dim: {}".format(n_dim))
n_classes = len(np.unique(tr_labels)) # number of classes in training set (how many categories are we classifying)
print("num classes: {}".format(n_classes))
n_hidden_units_one = 280 # ADJUST ACCORDINGLY
n_hidden_units_two = 300 # ADJUST ACCORDINGLY
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01 # ADJUST ACCORDINGLY (how quickly the models learns)

# EXPLANATION: create placeholders which tensorflow will fill with data
X = tf.placeholder(tf.float32,[None,n_dim]) # type = float32, None = any num rows, n_dim (or # features) cols of data
print("X: {}, {}".format(None, n_dim))
Y = tf.placeholder(tf.float32,[None,n_classes]) # type = float32, None = any num rows, n_classes cols of data
print("Y: {}, {}".format(None, n_classes))

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
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))

    accuracy_output = sess.run(accuracy, feed_dict={X: ts_features,Y: ts_labels})
    print("Test accuracy: {0:.3f}".format(accuracy_output))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
max_cost = np.max(cost_history)

print(max_cost)
if max_cost is None or max_cost == math.inf:
    max_cost = 10

plt.axis([0,training_epochs,0,max_cost])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print("F-Score:", round(f,3))
