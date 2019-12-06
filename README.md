# EDG Sound Classification Process

The following describes everything that you'll need to know to start classifying sound data using Tensorflow (prior coding experience preferable, but not necessary!). Please email kafkaloff@berkeley.edu if you have any questions.

## Getting Started

These instructions will help you download all necessary software and files to start running the python template on your local machine for development.

### Installations

Necessary files and software that you need to install/download and how to install/download them.

#### Command Line Installations

Run the following commands in your terminal:

```
pip install tensorflow
pip install librosa
```

#### Downloads

* [Download Python 3](https://www.python.org/downloads/)
* [Download sound_classification_template.py](https://github.com/kris10akemi/edg-sounds-tensorflow) (click on "Clone or Download")
* [Download sample sound sets (includes all lab data)](https://drive.google.com/drive/folders/1K8jmX2Bl_KinQgut0tyoxekdD_9Vonqo?usp=sharing)

### Running Python Files

To run a python file (*.py), enter the following in your terminal from the same directory as the python file:

```
python3 <file name>.py
```

## Training with Sample Sound Data

To get started, you need to have the sound_classification_template.py and sample sound sets (under Downloads) in the same directory. From there, you can run the following (processing the sound data will take around 5-10 min):

```
python3 sound_classification_template.py
```

Expect to see a graph and data similar to the following:

```
Class 0: 
	 accuracy: 1.000000 
	 most confused with: None
Class 1: 
	 accuracy: 0.000000 
	 most confused with: 0
Class 2: 
	 accuracy: 0.500000 
	 most confused with: 0
```
<img src="https://github.com/kris10akemi/edg-sounds-tensorflow/blob/master/sample_sound_sets_graph.png" width="400">

The accuracy and the class that each class confused its test data with the most are displayed by visualize_results.

To play around with more data, you can download the [Urban Sound Data set](https://urbansounddataset.weebly.com/). 

## Training Your Neural Network

### Template Configuration

#### Neural Network Configuration

Anything in sound_classification_template.py

#### Sound File Configuration

File organization is important when extracting sound data. For each class, there should training data and testing data. The training data is used to train our model and the testing data is used to measure how well our model classifies data based on the trained model. For example, if we have sound data for a dog barking. We should have two folders: dog bark training data (stored in "dog_bark_training" in this example) and dog bark testing data (stored in "dog_bark_testing" in this example). 

Things to update:
* **parent_dir** - refers to the parent directory
	- parent directory is the file path preceding the folder in the sub directories (or "sub dirs")
	- for example, from the location of our sound_classification_template.py, the "dog_bark_training" folder is actually stored in another folder 'sample_sound_sets_large', so when the code tries to extract sound data from there, we need to specify that parent_dir='sample_sound_sets_large' and it will extract from 'sample_sound_sets_large/dog_bark_training'
* **tr_sub_dir_labels** - specifies your training data, each element is the name of the directory pointing to the training data for a certain class and the number refers to an arbitrary numeric identifier for that class
* **ts_sub_dir_labels** - specifies your testing data, each element is the name of the directory pointing to the testing data for a certain class and the number refers to an arbitrary numeric identifier for that class (which should correspond to the training data numeric identifiers)

Notes:
* directory and folder are being used interchangeabley
* if the folder is in your current directory, you can set parent_dir = '.'
* if the folder is back a directory, you can set parent_dir = "../\<rest of file path from one directory back\>"

Example (6 training sets, 6 test sets, 6 class types):
```
parent_dir = 'sample_sound_sets_large'

tr_sub_dirs_labels = [["air_conditioner_training",0],
                      ["car_horn_training",1],
                      ["children_playing_training",2],
                      ["dog_bark_training", 3],
                      ["drilling_training", 4],
                      ["engine_idling_training", 5]] # training set
ts_sub_dirs_labels = [["air_conditioner_testing",0],
                      ["car_horn_testing",1],
                      ["children_playing_testing",2],
                      ["dog_bark_testing", 3],
                      ["drilling_testing", 4],
                      ["engine_idling_testing", 5]] # test set
```

### Formatting/Preparing Your Sound Data

Sound data needs to be in \*.wav format. You may need to look up software to convert your audio files, such as:
* iMovie: need to specify "Audio Only" and .wav when saving; this software is good for splitting up longer clips
* [Movavi Audio Converter](https://www.movavi.com/audioconvertermac/?utm_expid=.Dd9UbW3MR5KXUxx-Hzzlcg.0&utm_referrer=https%3A%2F%2Fwww.google.com%2F): legitimate, but limited acess and partial audio conversion
* [Online-Convert](https://audio.online-convert.com/convert-to-wav): less legitimate, but full audio conversion

Ideally your sound data should be trimmed and cleaned to avoid extraneous or irrelevant sound in the training data. 

## Understanding the ML Sound Classification Process

This is a step-by-step walk-through of the code to help any user understand what the code is doing.

### Extracting Sound Features 

Sounds vary drastically, even within the same category (e.g. laughter from one person sounds different from another, but they should both be classified as laughter). Thus, we use features of sounds to define each sound sample as features are used to represent relative similarities. The more similar the features of two sounds are, the more likely they are part of the same category.

There is room for improvement here dependong on the sounds that we are classifying. We can define our sounds by more, less, or different features than the ones in the template. These functions are from the "librosa" library, but there are other libraries and features within librosa available. Refer to the following for more on librosa: [Librosa Feature Extraction](https://librosa.github.io/librosa/feature.html).

Credit to [Aaqib Saeed](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/) for the following descriptions:
- **melspectrogram:** Compute a Mel-scaled power spectrogram
- **mfcc:** Mel-frequency cepstral coefficients
- **chorma-stft:** Compute a chromagram from a waveform or power spectrogram
- **spectral_contrast:** Compute spectral contrast, using method defined in [1]
- **tonnetz:** Computes the tonal centroid features (tonnetz), following the method of [2]

```
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
```

### Neural Net Configuration


- **training_epochs (ADJUSTABLE):** the number of times that the gradient descent optimizer sees ALL of the training data
- **n_dim:** the number of features that we are using to define our sound data
- **n_classes:** number of classes in our training set (how many categories we are classifying)
- **learning_rate (ADJUSTABLE):** how quickly our model learns from training data (amount of weight given to each sample)

```
training_epochs = 50
n_dim = tr_features.shape[1]
n_classes = len(np.unique(tr_labels))
n_hidden_units_one = 280 # TODO: ADJUST ACCORDINGLY
n_hidden_units_two = 300 # TODO: ADJUST ACCORDINGLY
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01 # TODO: ADJUST ACCORDINGLY
```

### Gradient Descent Optimizer

We use gradient descent to cluster similar sounds based on their features. Using derivatives, gradient descent moves in a direction that minimizes the cost as defined by cost_function.

**cost_function:** uses cross entropy where Y is binary indicator (whether or not prediction is correct) and y_ represents the predicted probability that that item is of correct class
* lower predicted prob -> smaller log(y_) -> increased cost (because negated)
* higher predicted prob -> larger log(y_) -> increased cost (because negated)

```
cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

### Prediction Model

The following is the process of training the gradient optimizer for our model. The optimizer is trained based on all of the training data for a number of training_epochs, improving at each iteration. In the same session, we use the training model to make predictions about our test set (ts_features). From there, we compute the accuracy of the model's performance by comparing the prediction (y_pred) to the actual results (ts_labels).

- **cost_history:** tracks the cost of each training epoch to plot the cost improvement later

```
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
```

### Visualizing Results

Displays the following:
- **accuracy per test:** how well the model performed on the test set for each class
- **most confused with:** the class that another class' test set was confused with the most
- **cost history plot:** how well the model improved over time

Please email kafkaloff@berkeley.edu to request further visualization and data features. 




# Experimenting with Coral Gripper Events

## Mission

When a gripper successfully grasps onto coral and loses grip, unique sounds are generated. Grip loss is accidental, unlike an intentional grip release where the gripper mouth opens. Grip loss is unpredictable and usually leaves the ROV in an unstable state, thus the ROV should always reset prior to re-attempting or giving up. If we use sounds to predict a successful coral grasp or grip loss, we can partially automate the grasping process by predicting the coral gripper's state. 

## Lab Experiments

### Current Progress

There are currently two states that we are trying to distinguish: successful grip and grip loss. This is being tested in the lab by pushing the gripper onto the edge of Monica's pottery cup to simulate successful grip and then pulling it back off (without opening) to simulate grip loss. An iPhone is being used to record and iMovie is being used to split the sound clips into two classes with training and testing sets (.wav files). The current data has been uploaded and it is in the "coral_gripper" folder.

This is mostly in the experimental phase right now and we aren't sure if sounds will be useful in making these event classifications, but given that vision is a manual operation and it is sometimes obstructed underwater, sounds could be helpful.

### ML Training Considerations

* Sound features are averaged over the duration of the clip, so any extraneous silence on the ends can affect the data
* Volume may be a useful feature to consider adding
* Collect more testing + training data
* Adjust parameters (as specified in Neural Net Configuration)

### Real-Life Application Considerations

* Sounds are less audible underwater (may need a hydrophone)
* There will be far more extraneous noise in the ocean


## Acknowledgments

* [Aaqib Saeed](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)
* [Urban Sound](https://urbansounddataset.weebly.com/)
