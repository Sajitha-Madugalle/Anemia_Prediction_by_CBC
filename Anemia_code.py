import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dirfile=pd.read_csv('anemia.csv')

x_train = dirfile.values[:1000,:5]
y_train = dirfile.values[:1000,5]

x_test = dirfile.values[1000:,:5]
y_test = dirfile.values[1000:,5]

model=keras.Sequential([
    keras.layers.Dense(2, input_shape=(5,),activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=500)
model.evaluate(x_test,y_test)


results=model.predict(x_test)



print(np.argmax(results[7]))
print(y_test[7])
