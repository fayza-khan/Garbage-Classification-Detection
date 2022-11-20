# Objective 

Garbage detection is an image classification model that aims to classify images of 4 types of waste into their corresponding biodegradable categories. This model is capable of classifying images into food, carboard, metal and glass waste. While it is unrealistic that a user would only have images of these 4 types of waste, this project aims to create a proof-of-concept model to demonstrate deployment of deep learning image classification models.

# Source of data

The data used for this project was sourced from:

  - using Google Images and 'Download All Images' extension, and,
  - using an image dataset collected manually by Gary Thung and Mindy Yang.

The entire data consists of 2,048 images for food, carboard, metal and glass categories. The data is then cleaned by removing all images that might not open in python due to corrupted, mislabelled, or incorrect extension applied, which leads to a total of 1,735 images. The data is then converted into batches using tensorlow pipeline for processing, building a total of 55 batches of 32 images each.

Here are a few example images from each class:

<img src="https://user-images.githubusercontent.com/70770111/202913115-80ceb4c1-6681-47f5-9d27-3f1dfeb8caee.jpg" width="200"> <img src="https://user-images.githubusercontent.com/70770111/202913124-6feb6fab-70f5-4f4f-b4e1-57e54f6276f2.jpeg" width="200"> <img src="https://user-images.githubusercontent.com/70770111/202913129-2aa2f38a-7f44-45bc-83fc-2cc453abcf1b.jpg" width="200"> <img src="https://user-images.githubusercontent.com/70770111/202913136-36d293bb-9546-4646-a6cb-fb53e086f8b2.jpg" width="200">

One tricky thing about this classification will be getting our model to distinguish between objects that might contain more than one category, say a utensil made of glass and metal.

# Modelling

### Approach

The approach was to build a simple neural network model using convolutional neural networks (CNNs) and a basic keras Sequential model. In order to achieve higher accuracy, the hyperparameters, including, optimizer, filters, epochs, and others, were experimented with different values. 

For instance,

##### - Optimizer

1. Stochastic gradient descent (SGD): Using this, the model achieved an accuracy of 96% for testing data, with precision and recall both coming equal to 0.98. A total of 40 epochs were used to fit the training data in the model.

<img width="300" alt="Accuracy_SGD" src="https://user-images.githubusercontent.com/70770111/202916942-80a39059-7776-4984-9273-1c6b340ed7d4.PNG"> <img width="300" alt="Loss_SGD" src="https://user-images.githubusercontent.com/70770111/202916951-7dcc88c6-38ac-41dd-b1e6-6db86603cac2.PNG">


2. Adam: Using this, the model achieved an accuracy of 99% for testing data, with precision and recall both coming equal to 1. A total of 20 epochs were used to fit the training data in the model.

<img width="300" alt="Adam_accuracy" src="https://user-images.githubusercontent.com/70770111/202919728-4b6a1c89-7d76-45a6-8730-0301413fd7e2.PNG"> <img width="300" alt="Loss_Adam" src="https://user-images.githubusercontent.com/70770111/202920133-8e27fbb9-f615-4697-b743-25451742bcf5.PNG">


Using the above results, adam was used as the optimizer.

Similarly, filters and epochs were modified to get different results with the aim to get better accuracy and to avoid overfitting at the same time.

### Metrics
My aim for this model is that it classifies images as accurately as possible. There is no benefit towards seeing less false negatives over false positives and vice versa since both scenarios result in a misclassification. For this reason, I choose to optimize for accuracy.

### Success Criteria
For this model, I aim to create a model that performs at, at least 90% accuracy and that also shows little-to-no overfitting.

# Tools / Techniques used: 
- Programming Language:
  - Python 3
- Technique:
  - Deep learning using CNN
- Major libraries:
  - Tensorflow
  - Keras
  - Matplotlib
  - Scikit-Learn
  
# Future Developments

The next stages of this project will include further model improvement, such as,
  - could add a probability threshold, that an image is only included in a particular album if it is '80% sure' that that image is of that class (for example). This might also help in handling the tricky issues of an image of 'Glass and metal bottle', as we can see 1 of the metal case was predicted as glass in test dataset. 
  <img width="300" alt="1" src="https://user-images.githubusercontent.com/70770111/202920038-f495f00e-6586-411b-8c90-5f077b0f0d62.PNG">

  
  - the next iteration of deployment would be to have an 'other' album, so that images of all kinds can be input into the model and the model will place images that are not one of the 4 classes into the 'other' album. 
