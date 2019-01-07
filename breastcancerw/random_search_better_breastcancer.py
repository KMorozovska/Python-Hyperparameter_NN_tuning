from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
import time

secure_random = random.SystemRandom()
ilosc_iteracji_random_search = 40

class MyParams(object):
    
   def __init__(self, lr, input_n_n, layers_num, accuracy, logloss):
        self.lr = lr
        self.input_n_n = input_n_n
        self.layers_num = layers_num
        self.accuracy = accuracy
        self.logloss = logloss

# TODO : toString() ??!

#import data
dataset = pd.read_csv("/home/kasia/Dokumenty/Studia/INŻ/ml/breastcancerw/data.csv", delimiter=",")
dataset.diagnosis.unique()
dataset['diagnosis'] = dataset['diagnosis'].map({'M':1,'B':0})

Y=np.asarray(dataset['diagnosis'].to_frame())
X=np.asarray(dataset.drop(['diagnosis'], axis=1, inplace=False))

X1 = np.delete(X,31,1)

X_normalized = preprocessing.scale(X1)

print('X1')
print(X1)
print('znormalizowane funkcją scale()')
print(X_normalized)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=1)


scores_table = [] #lista w ktorej sa zapisywane obiekty: parametry+wyniki

bestModel = MyParams(0,0,0,0,0)

start_time = time.time()

for x in range (0,ilosc_iteracji_random_search):
    #do everything
    print ('\n', x+1, ' iteration')
    #set parameters
    input_neurons_number=range(30, 80,5)
    layers_number=range(1,5)
    learning_rate=[i*0.1 for i in range(10)]

    #random select parameters
    chosen_lr = secure_random.choice(learning_rate)
    print('Learning rate: ', chosen_lr)                         #wypisanie na ekran zeby porownac czy wszystko ok
    chosen_neurons = secure_random.choice(input_neurons_number)
    print('Neurons number: ', chosen_neurons)
    chosen_layers = secure_random.choice(layers_number)
    print('Layers number: ', chosen_layers, "\n")
    
    #create model
    model = Sequential()
    model.add(Dense(chosen_neurons, input_dim=31, activation='relu'))   #warstwa wejsciowa
    for j in range (0, chosen_layers):
        model.add(Dense(chosen_neurons//2, activation='relu'))           #wewnetrzne warstwy
    model.add(Dense(1,activation='sigmoid'))                            #warstwa wyjsciowa
    
    #compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    sgd = optimizers.SGD(lr=chosen_lr, clipnorm=1.)
    #fit the model
    history=model.fit(X_train,y_train,batch_size=31, epochs=10, verbose=0)
    
    #evaluate
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    if(scores[1]*100 > bestModel.accuracy):
        bestModel = MyParams(chosen_lr, chosen_neurons, chosen_layers, scores[1]*100, scores[0]*100)
    del model

#rysuje wykres dla modelu ktory wyszedl najlepiej: not yet

#wypisanie najlepszego modelu
print("To wybralismy: ", bestModel.accuracy, bestModel.lr, bestModel.input_n_n, bestModel.layers_num)
print('\n\n')

elapsed_time = time.time() - start_time

print("Czas jaki minal na ", ilosc_iteracji_random_search, " iteracji: ", elapsed_time)

# draw histogram ! 

scores_table_new = np.hstack((scores_table))

print(scores_table)
print(scores_table_new)

plt.hist(scores_table_new, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram wynikow 60 iteracji random search")
plt.show()

scores_table_new_2 = np.hstack((scores_table))

plt.hist(scores_table_new_2, bins=12)  # arguments are passed to np.histogram
plt.title("Histogram wynikow 60 iteracji random search")
plt.show()