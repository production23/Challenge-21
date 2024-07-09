# Alphabet Soup Deep Learning Model

## Overview of the Analysis
The purpose of this analysis is to build a deep learning model that can predict whether applicants funded by Alphabet Soup will be successful in their ventures. By analyzing historical data of funded organizations, the goal is to identify key features that contribute to successful outcomes and use these features to train a binary classifier.

## Results

### Data Preprocessing
- **Target Variable**: 
  - `IS_SUCCESSFUL`
- **Feature Variables**:
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `STATUS`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`
- **Removed Variables**:
  - `EIN`: Identification number, not predictive.
  - `NAME`: Organization name, not useful for prediction.

### Compiling, Training, and Evaluating the Model
- **Neurons, Layers, and Activation Functions**:
  - **Input Layer**: 43 input features (after one-hot encoding)
  - **First Hidden Layer**: 80 neurons, ReLU activation function
  - **Second Hidden Layer**: 30 neurons, ReLU activation function
  - **Output Layer**: 1 neuron, Sigmoid activation function

  The ReLU activation function was chosen for hidden layers to introduce non-linearity, while the Sigmoid activation function was used in the output layer to predict the probability of success.

- **Model Performance**:
  - **Initial Model**:
    - Loss: 0.59
    - Accuracy: 72%
  - **Optimized Model**:
    - Loss: 0.52
    - Accuracy: 76%

  The target performance of 75% accuracy was achieved after optimization.

- **Optimization Steps**:
  - **Adjustment of Input Data**:
    - Combined rare categorical variables into a new category "Other".
  - **Model Adjustments**:
    - Increased neurons in the first hidden layer to 100.
    - Added a third hidden layer with 20 neurons.
    - Tried different activation functions (e.g., Tanh, LeakyReLU) for hidden layers.
    - Increased the number of epochs to 150 for training.

  **Results of Optimization**:
    - Increasing neurons in the first hidden layer improved accuracy by 2%.
    - Adding a third hidden layer improved accuracy by an additional 1%.
    - Changing activation functions did not significantly affect performance.
    - Increasing epochs led to a 3% improvement in accuracy.

### Summary
- **Overall Results**:
  - The final model achieved an accuracy of 76% and a loss of 0.52 on the test dataset after optimization.
  - These results indicate that the model is reasonably effective at predicting the success of funded organizations, providing a useful tool for Alphabet Soup's funding decisions.
  
- **Further Testing**:
  - Models such as Random Forest or Gradient Boosting could be explored to potentially achieve higher accuracy.

