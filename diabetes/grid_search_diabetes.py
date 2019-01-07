from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
import time

class MyParams(object):
    
   def __init__(self, lr, input_n_n, layers_num, accuracy, logloss):
        self.lr = lr
        self.input_n_n = input_n_n
        self.layers_num = layers_num
        self.accuracy = accuracy
        self.logloss = logloss


#set parameters
min_input_neurons_number=100
max_input_neurons_number=150
min_layers_number=1
max_layers_number=5
min_learning_rate=1
max_learning_rate=10

#import data
dataset_data = pd.read_csv("/home/kasia/Dokumenty/Studia/INŻ/ml/diabetes/data.csv", delimiter=",", header=None, usecols=[0,1,2,3,4,5,6,7])
dataset_output = pd.read_csv("/home/kasia/Dokumenty/Studia/INŻ/ml/diabetes/data.csv", delimiter=",", header=None, usecols=[8])


X=np.asarray(dataset_data)
Y=np.asarray(dataset_output)

X_normalized = preprocessing.scale(X)

print('X')
print(X)
print('znormalizowane funkcją scale()')
print(X_normalized)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=1)



scores_table = []

bestModel = MyParams(0,0,0,0,0)
start_time = time.time()

for x in range (min_input_neurons_number,max_input_neurons_number+1,5):                #grid search
    #do everything
    #print ('\n', x, ' input neurons number')
    for y in range (min_layers_number, max_layers_number+1):
        #print (y, ' layers number')
        for z in range (min_learning_rate, max_learning_rate+1):
            #print (z, ' learning rate')
            
            #x,y,z to wybrane parametry
            chosen_lr = z/10
            print('\n', 'Learning rate: ', chosen_lr)                         #wypisanie na ekran zeby porownac czy wszystko ok
            chosen_neurons = x
            print('Neurons number: ', chosen_neurons)
            chosen_layers = y
            print('Layers number: ', chosen_layers, "\n")
            
            #create model
            model = Sequential()
            model.add(Dense(chosen_neurons, input_dim=8, activation='relu'))   #warstwa wejsciowa
            for j in range (0, chosen_layers):
                model.add(Dense(chosen_neurons//2, activation='relu'))           #wewnetrzne warstwy
            model.add(Dense(1,activation='sigmoid'))                            #warstwa wyjsciowa
    
            #compile model
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            sgd = optimizers.SGD(lr=chosen_lr, clipnorm=1.)
            #fit the model
            history=model.fit(X_train,y_train,batch_size=12, epochs=10, verbose=0)
    
            #evaluate
            scores = model.evaluate(X_test, y_test)
            
            print('\n')
    
            scores_table.append(scores[1]*100)
    
            if(scores[1]*100 > bestModel.accuracy):
                bestModel = MyParams(chosen_lr, chosen_neurons, chosen_layers, scores[1]*100, scores[0]*100)
                
            del model
    

#rysuje wykres dla modelu ktory wyszedl najlepiej: not yet

#wypisanie najlepszego modelu
print("To wybralismy: ", bestModel.accuracy, bestModel.lr, bestModel.input_n_n, bestModel.layers_num)
print('\n\n')

elapsed_time = time.time() - start_time

print("Czas jaki minal na wykonanie algorytmu ", elapsed_time)

# TODO : rysowac wykres najlepszego modelu





# draw histogram ! 

scores_table_new = np.hstack((scores_table))

plt.hist(scores_table_new, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram wynikow 60 iteracji random search")
plt.show()

scores_table_new_2 = np.hstack((scores_table))

plt.hist(scores_table_new_2, bins=12)  # arguments are passed to np.histogram
plt.title("Histogram wynikow 60 iteracji random search")
plt.show()