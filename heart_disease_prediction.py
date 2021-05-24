import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df = pd.read_csv("heart.csv")
print(df.head())

df.columns = ['age', 'sex_male', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
              'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope',
              'num_major_vessels', 'thalassemia', 'target']

print(df.head())

model = LinearSVC()

X = df.drop('target',axis=1)
y = df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)
    
model.fit(X_train,y_train)    # model.fit(X_train,y_train)
prediction = model.predict(X_test)   # model.predict(X_test)

import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('heart_prediction.tflite', 'wb') as f:
  f.write(tflite_model)


print(confusion_matrix(y_test,prediction))

