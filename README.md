
# Alphabet Soup Charity Analysis

## Overview

The purpose of this analysis is to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. This model is built using deep learning techniques with TensorFlow and Keras.

## Data Preprocessing

1. **Data Source:** The dataset used is `charity_data.csv`, which contains various features of organizations funded by Alphabet Soup.
2. **Target Variable:** `IS_SUCCESSFUL`
3. **Feature Variables:** All columns except `EIN`, `NAME`, and `IS_SUCCESSFUL`
4. **Dropped Columns:** `EIN`, `NAME`

### Steps

1. **Load and Inspect Data:**
   ```python
   import pandas as pd
   application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
   application_df.head()
   ```

2. **Drop Non-beneficial Columns:**
   ```python
   application_df = application_df.drop(columns=['EIN', 'NAME'])
   ```

3. **Determine Unique Values:**
   ```python
   unique_counts = application_df.nunique()
   print(unique_counts)
   ```

4. **Replace Rare Categories in `APPLICATION_TYPE`:**
   ```python
   application_type_counts = application_df['APPLICATION_TYPE'].value_counts()
   replace_application_types = application_type_counts[application_type_counts < 100].index
   application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(replace_application_types, 'Other')
   ```

5. **Replace Rare Categories in `CLASSIFICATION`:**
   ```python
   classification_counts = application_df['CLASSIFICATION'].value_counts()
   classifications_to_replace = classification_counts[classification_counts < 100].index.tolist()
   application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(classifications_to_replace, 'Other')
   ```

6. **Convert Categorical Data to Numeric:**
   ```python
   application_df = pd.get_dummies(application_df)
   ```

7. **Split Data into Features and Target:**
   ```python
   X = application_df.drop(columns=['IS_SUCCESSFUL'])
   y = application_df['IS_SUCCESSFUL']
   ```

8. **Split Data into Training and Testing Sets:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

9. **Scale the Data:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

## Model Building

1. **Define the Model:**
   ```python
   import tensorflow as tf
   nn = tf.keras.models.Sequential()
   nn.add(tf.keras.layers.Dense(units=80, activation='relu', input_dim=X_train_scaled.shape[1]))
   nn.add(tf.keras.layers.Dense(units=30, activation='relu'))
   nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
   nn.summary()
   ```

2. **Compile the Model:**
   ```python
   nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   ```

3. **Train the Model:**
   ```python
   history = nn.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
   ```

## Model Evaluation

1. **Evaluate the Model:**
   ```python
   loss, accuracy = nn.evaluate(X_test_scaled, y_test)
   print(f'Model Loss: {loss}, Model Accuracy: {accuracy}')
   ```

## Export the Model

1. **Save the Model:**
   ```python
   nn.save('AlphabetSoupCharity.h5')
   ```

## Summary

The model was successfully built and trained to predict the success of Alphabet Soup-funded organizations. Further optimization and tuning can be done to improve the model's performance.
