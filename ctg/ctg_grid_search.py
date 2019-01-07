from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


class MyParams(object):
    
   def __init__(self, lr, input_n_n, layers_num, accuracy, logloss):
        self.lr = lr
        self.input_n_n = input_n_n
        self.layers_num = layers_num
        self.accuracy = accuracy
        self.logloss = logloss


#set parameters
min_input_neurons_number=10
max_input_neurons_number=70
min_layers_number=1
max_layers_number=5
min_learning_rate=1
max_learning_rate=10

seed = 7
np.random.seed(seed)

#import data
dataset = pd.read_excel("/home/kasia/Dokumenty/Studia/INŻ/ml/diabetes/CTG.xls", sheetname="Raw Data")


Y=np.asarray(dataset['NSP'].to_frame())
X=np.asarray(dataset.drop(['NSP'], axis=1, inplace=False))


X1 = np.delete(X,[0,1,2,38],1)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


print("------------------------")
print(X1)
print(Y)


X_normalized = preprocessing.scale(X1)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, dummy_y, test_size=0.3, random_state=1)


scores_table = [] #lista w ktorej sa zapisywane obiekty: parametry+wyniki
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
            model.add(Dense(chosen_neurons, input_dim=35, activation='relu'))   #warstwa wejsciowa
            for j in range (0, chosen_layers):
                model.add(Dense(chosen_neurons//2, activation='relu'))           #wewnetrzne warstwy
            model.add(Dense(3,activation='softmax'))                            #warstwa wyjsciowa
    
            #compile model
            model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
            sgd = optimizers.SGD(lr=chosen_lr, clipnorm=1.)
            #fit the model
            history=model.fit(X_train,y_train,batch_size=15, epochs=4, verbose=0)
    
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
plt.title("Histogram wynikow algorytmu grid search")
plt.show()
