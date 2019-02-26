# EDG Sound Classification Process

The following describes everything that you'll need to know to start classifying sound data using Tensorflow with no prior coding experience necessary. Please email kafkaloff@berkeley.edu if you have any questions.

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
* [Download sample sound sets](https://drive.google.com/drive/folders/14DmBB15mLApoZCy9pjt0y1hjFr-XVZ-H?usp=sharing)

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

### Formatting/Preparing Your Sound Data

## Understanding the ML Sound Classification Process

This is a step-by-step walk-through of the code to help any user understand what the code is doing.

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

**cost_function:**
- uses cross entropy where Y is binary indicator (whether or not prediction is correct) and y_ represents the predicted probability that that item is of correct class
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

## Acknowledgments

* [Aaqib Saeed](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)
* [Urban Sound](https://urbansounddataset.weebly.com/)
