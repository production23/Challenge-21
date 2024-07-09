# Alphabet Soup Charity Funding Predictor

## Overview
This project aims to build a deep learning model to predict the success of applicants funded by Alphabet Soup. By analyzing historical data of funded organizations, the goal is to identify key features that contribute to successful outcomes and use these features to train a binary classifier.

## Data Preprocessing
- **Target Variable**: `IS_SUCCESSFUL`
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
- **Removed Variables**: `EIN` and `NAME`

## Model Compilation, Training, and Evaluation
### Model Architecture
- **Input Layer**: 43 input features (after one-hot encoding)
- **First Hidden Layer**: 80 neurons, ReLU activation function
- **Second Hidden Layer**: 30 neurons, ReLU activation function
- **Output Layer**: 1 neuron, Sigmoid activation function

### Training the Model
The model was trained using 100 epochs with a validation split of 20%.

### Evaluating the Model
The model was evaluated on a test dataset, achieving the following performance:
- **Initial Model**:
  - Loss: 0.59
  - Accuracy: 72%
- **Optimized Model**:
  - Loss: 0.52
  - Accuracy: 76%

### Optimization Steps
1. Combined rare categorical variables into a new category "Other".
2. Increased neurons in the first hidden layer to 100.
3. Added a third hidden layer with 20 neurons.
4. Tried different activation functions (e.g., Tanh, LeakyReLU) for hidden layers.
5. Increased the number of epochs to 150 for training.

## Results
The final model achieved an accuracy of 76% and a loss of 0.52 on the test dataset after optimization. These results indicate that the model is reasonably effective at predicting the success of funded organizations.

## Recommendations
For potentially higher accuracy, alternative models such as Random Forest or Gradient Boosting could be explored. These models can handle non-linear relationships and interactions between features more effectively.

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Starter_Code.ipynb
   ```

4. **Train and Evaluate the Model**:
   Follow the steps in the Jupyter Notebook to preprocess the data, compile, train, and evaluate the model.

5. **Save the Model**:
   The model will be saved as `AlphabetSoupCharity.h5` in the working directory.

6. **Download the Saved Model**:
   ```python
   from google.colab import files
   files.download("AlphabetSoupCharity.h5")
   ```
