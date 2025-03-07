import tensorflow as tf
import matplotlib.pyplot as plt 

from Preprocessing.preprocessing import preprocessing
from Models.cnn import create_CNN_model
from Models.transferLearning import create_MobileNet_model


dict_train = preprocessing()
dict_val = preprocessing(_type="validation")
dict_test = preprocessing(_type="test")

X_train, y_train = dict_train['feature'], dict_train['label']
X_val, y_val = dict_val['feature'], dict_val['label']
X_test, y_test = dict_test['feature'], dict_test['label']


CNN_model = create_CNN_model()

# Scheduler: ReduceLROnPlateau
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',          # Monitors validation loss
    factor=0.5,                  # Reduces learning rate by half
    patience=5,                  # Number of epochs without improvement before reduction
    min_lr=1e-6,                 # Lower limit for learning rate
    verbose=1                    # Displays learning rate changes
)

# CNN_model.summary()

outputs = CNN_model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, verbose='auto', validation_data=(X_val, y_val), shuffle=True, callbacks=[lr_scheduler])

plt.figure(figsize=(6,5))
plt.plot(outputs.epoch, outputs.history["loss"], 'g', label='Training loss')
plt.plot(outputs.epoch, outputs.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.title('Reduction of the cost function')
plt.legend()
plt.show()
