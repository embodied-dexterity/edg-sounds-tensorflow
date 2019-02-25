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

Currently, the accuracy and the class that each class confused its test data with the most are displayed by visualize_results. Please email kafkaloff@berkeley.edu to request further visualization and data features. 

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

## Acknowledgments

* [Aaqib Saeed](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)
* [Urban Sound](https://urbansounddataset.weebly.com/)
