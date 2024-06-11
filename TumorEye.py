#Import the dataset.
#[Click here for the dataset](https://gist.github.com/KhanradCoder/35a6beea49e5b9ba62797e595a9626c0)
import pandas as pd
dataset = pd.read_csv('cancer_cropped_dataset.csv')
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

#Split the data into a training set and a testing set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Build and train the model.
import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

#Evaluate the model.
model.evaluate(x_test, y_test)

#Make predictions
candidates = pd.read_csv('TumorEye_Predict.csv')
candidateList = candidates.drop(columns=["Candidate"])
prediction = model.predict(candidateList)
print("")
print("Making Predictions")
print("0.00 to 0.50: probably not malicious")
print("0.51 to 1.00: probably malicious")
print("One dataset per line:")
print(prediction)