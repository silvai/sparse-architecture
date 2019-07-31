import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

file_A = "../data/data_A.csv"
real_A = pd.read_csv(file_A, sep=',', header=None, low_memory=False)

filename = "../data/testData.csv"
trainDataSet = pd.read_csv(filename, sep=',', header=None, low_memory=False)
X = trainDataSet.values[0:1000, :]
Y = trainDataSet.values[1000:, :]

norms = []
splits = []
split_size = 0.35
splits.append(split_size * len(X))
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=split_size)

# loss_vals = []
# accuracy_vals = []
# training_accuracy = []
# validation_accuracy = []
# test_accuracy = []
# width_vals = []
# width = int(layer_size * 0.4)
layer_size = train_x.shape[1]
num_epochs = 1000
# input_layer = Input((layer_size,))
# # encode_layer0  = Dense(int(layer_size * .9), activation='relu')(input_layer)
# # encode_layer05 = Dense(int(layer_size * .8), activation='relu')(encode_layer0)
# # encode_layer1  = Dense(int(layer_size * .7), activation='relu')(encode_layer05)
# # encode_layer15 = Dense(int(layer_size * .6), activation='relu')(encode_layer1)
# # encode_layer2  = Dense(int(layer_size * .5), activation='relu')(encode_layer15)
# # encode_layer3  = Dense(int(layer_size * .4), activation='relu')(encode_layer2)
# #
# # bottleneck     = Dense(width, activation='sigmoid')(encode_layer3)
# #
# # decode_layer1  = Dense(int(layer_size * .4), activation='relu')(bottleneck)
# # decode_layer2  = Dense(int(layer_size * .5), activation='relu')(decode_layer1)
# # decode_layer25 = Dense(int(layer_size * .6), activation='relu')(decode_layer2)
# # decode_layer3  = Dense(int(layer_size * .7), activation='relu')(decode_layer25)
# # decode_layer35 = Dense(int(layer_size * .8), activation='relu')(decode_layer3)
# # decode_layer4  = Dense(int(layer_size * .9), activation='relu')(decode_layer35)
# #
# # output_layer   = Dense(layer_size)(decode_layer4)
# h_layer1 = Dense(int(layer_size * .5))(input_layer)
# output_layer = Dense((layer_size))(h_layer1)

# model = Model(inputs=input_layer, outputs=output_layer)

model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='relu'))
model.add(Dense(5))

sgd = optimizers.SGD(lr=0.001)
model.compile(optimizer=sgd, loss='mean_squared_error')
fit_mod = model.fit(train_x, train_y, epochs=num_epochs, batch_size=4, validation_data=(test_x, test_y))

estimated_output = model.predict(train_x[0:5])
a_hat = np.identity(5)
a_hat = model.predict(a_hat)
difference_a = real_A - a_hat
print("--------------------Using Identity----------------------")
print("-------------------------------------------------------")
print("A_hat:\n", a_hat)
print("A:\n", real_A)
print("Difference:\n", difference_a)
print(np.linalg.norm(difference_a))
print("-------------------------------------------------------")

inverse_x = np.linalg.pinv(estimated_output)
a_hat_est = np.matmul(estimated_output, inverse_x)
difference_b = real_A - a_hat_est
print("--------------------Using Algebra----------------------")
print("-------------------------------------------------------")
print("real_A:\n", real_A)
print("A_hat_est:\n", a_hat_est)
print("Difference:\n", difference_b)
print(np.linalg.norm(difference_b))
print("-------------------------------------------------------")

# plt.plot(fit_mod.history['loss'])
# plt.plot(fit_mod.history['val_loss'])
# plt.title('Accuracy of A Reproduction')
# plt.ylabel('Loss')
# plt.xlabel('Number of Samples')
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.show()


# loss_vals.append(np.mean(min(fit_mod.history.values())))
# width_vals.append(width)
#
#     print("Autoencoder with bottleneck width of: ", width, "\nloss function: binary_crossentropy\nbatch size: 300\n"
#                                                            "model number: ", m, " of ", num_models)
# plt.title("Loss Values as Bottleneck Width Increases Trained With " + str(num_models) + " Models")
# plt.plot(width_vals, loss_vals)
# plt.ylabel("Mean Loss Values")
# plt.xlabel("Bottleneck Width")
# plt.show()


# plt.plot(fit_mod.history['loss'])
# plt.plot(fit_mod.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper right')
# plt.show()

