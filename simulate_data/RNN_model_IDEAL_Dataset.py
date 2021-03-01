def import_lib():
        import pandas as pd
        import csv
        from numpy import dstack, hstack
        from keras.utils import to_categorical
        from sklearn.model_selection import train_test_split
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Dropout
        from keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
        from keras.optimizers import SGD, Adam, RMSprop
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
        import tensorflow_addons as tfa
        import tensorflow as tf
        import matplotlib.pyplot as plt
        import seaborn as sns
        from numpy import mean
        from numpy import std
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import Dropout
        from keras.layers.convolutional import Conv1D
        from keras.layers.convolutional import MaxPooling1D
        from keras.layers.merge import concatenate
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        import time

def load_dataset():
def load_dataset():
        data_filenames = ['data_0_v5_3.csv','data_1_v5_3.csv','data_2_v5_3.csv','data_3_v5_3.csv','data_4_v5_3.csv','data_5_v5_3.csv','data_6_v5_3.csv','data_7_v5_3.csv'
        ,'data_8_v5_3.csv','data_9_v5_3.csv']
        photodiode_arr = ['6','1','2','3','9','8','7','4','5']
        data_frame = []
        for i,file in enumerate (data_filenames):
        data = pd.read_csv(file)
        data_frame.append (data)
        data_frame = pd.concat(data_frame, join = 'inner')
        data_3D = list()
        for i, photodiode_name in enumerate(photodiode_arr):
        data_3D.append(data_frame[photodiode_name].values.reshape(-1, ))  # sliding window is hard
        data_3D = dstack(data_3D)
        data_3D = data_3D.reshape((-1, 500,9))
        data_y = pd.read_csv('label_v5.csv')
        result_none = np.where(data_y == -1)
        data_y = data_y.drop([data_y.index[260],   data_y.index[891],   data_y.index[892]], axis = 0)
        data_3D = np.delete(data_3D, [260, 891, 892 ], axis = 0)
        data_y = to_categorical(data_y,num_classes=10)
        Xtra,Xval,Ytra,Yval = train_test_split(data_3D,data_y,test_size=0.2,shuffle=True)
        return Xtra, Ytra, Xval, Yval



def get_trainable_params(model):
  number_of_weights = [K.count_params(w) for w in model.trainable_weights]
  k = 0
  for layer in model.layers:
    print('_ _ _ '*10)
    print(layer.name)
    for weights in layer.trainable_weights:
      print(f'{weights.name}, \nshape={weights.shape} ===> {number_of_weights[k]}')
      k+=1
  print('_ _ _ '*10)
  total = np.array(number_of_weights).sum().astype(int)
  print(f'Total = {total}')
  return total
  
  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,figsize=(5,5)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    adapted from https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    
    print(cm)
    
    f, ax = plt.subplots(1,figsize=figsize)
    ax.imshow(cm, interpolation='nearest', cmap=cmap,)
    ax.set_title(title)
    # ax.setcolorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      ax.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
    
    f.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')        


def build_training_model():
        verbose, epochs, batch_size, num_Neurons = 0, 40, 128, 100
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        # model.add(GRU(150, input_shape=(n_timesteps,n_features), return_sequences=True))
        # model.add(GRU(50))
        model.add(GRU(80, input_shape=(n_timesteps,n_features), return_sequences=False))
        # model.add(LSTM(80))
        # model.add(Dropout(0.2))
        # model.add(GRU(150))
        # model.add(Bidirectional( GRU(100))) 
        # model.add(Dropout(0.5))
        # model.add(Dense(100, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.summary()

def evaluate_model(trainX, trainy, testX, testy):
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model_checkpoint= ModelCheckpoint('RNN_model_weights.hdf5', save_best_only=True, monitor='val_loss', 
                                  mode='auto', save_weights_only=True)
	# fit network
	history = model.fit(trainX, trainy, epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=verbose)
	# evaluate model
	# _, accuracy = model.evaluate(testX, testy,batch_size=batch_size, verbose=0)
	return accuracy, history

def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def run_experiment(repeats=10):
	trainX, trainy, testX, testy = load_dataset()
	scores = list()
	for r in range(repeats):
		score, history = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Training Loss VS Validation Loss')
		plt.ylabel('Loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
	summarize_results(scores)


def training_model():
        load_dataset()
        build_training_model()
        run_experiment()


def main():
        import_lib()
        training_model()
        get_real_time_data()
        predict_real_time_data()

def if __name__ == "__main__":
    main()
    pass
