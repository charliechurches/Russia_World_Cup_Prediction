import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2


# In[98]:


# Prepare the vectors to the appropiate shape
im_rows = 512
im_cols = 512
batch_size = 10
im_shape = (im_rows, im_cols,1)

dataset_path = 'data'
overview = pd.read_csv(dataset_path + "\results.csv")
ov_array = overview.as_matrix()
images = np.array([])


images = images.reshape(len(ov_array), *im_shape)
images.shape


# In[101]:


x_train = images[:, :] / 1024
y_train = ov_array[:, 2]
x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12345,
)
# Configure tensorboard
tensorboard = TensorBoard(
    log_dir=r'logs\{}'.format('cnn_1layer_medical'),
    write_graph=True,
    write_grads=True,
    histogram_freq=1,
    write_images=True,
)




# Define the model. In this case it is a CNN
cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=1, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])


# Compile the model
cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

# Fit the model
cnn_model.fit(
    x_train, y_train, batch_size=batch_size,
    epochs=2, verbose=1,
    validation_data=(x_validate, y_validate)
)


# In[107]:


# Get the score and print it
score = cnn_model.evaluate(x_validate, y_validate, verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))

