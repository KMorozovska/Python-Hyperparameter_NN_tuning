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

secure_random = random.SystemRandom()
ilosc_iteracji_random_search = 60

class MyParams(object):
    
   def __init__(self, lr, input_n_n, layers_num, accuracy, logloss):
        self.lr = lr
        self.input_n_n = input_n_n
        self.layers_num = layers_num
        self.accuracy = accuracy
        self.logloss = logloss

# TODO : toString() ??!

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

X_train, X_test, y_train, y_test = train_test_split(X_normalized, dummy_y, test_size=0.2, random_state=1)


scores_table = [] #lista w ktorej sa zapisywane obiekty: parametry+wyniki
bestModel = MyParams(0,0,0,0,0)
start_time = time.time()

for x in range (0,ilosc_iteracji_random_search):
    #do everything
    print ('\n', x+1, ' iteration')
    #set parameters
    input_neurons_number=range(50,100,5)
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
    model.add(Dense(chosen_neurons, input_dim=35, activation='relu'))   #warstwa wejsciowa
    for j in range (0, chosen_layers):
        model.add(Dense(chosen_neurons//2, activation='relu'))           #wewnetrzne warstwy
    model.add(Dense(3,activation='softmax'))                            #warstwa wyjsciowa
    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = optimizers.SGD(lr=chosen_lr, clipnorm=1.)
    
    #fit the model
    history=model.fit(X_train,y_train,batch_size=31, epochs=10, verbose=0)
    
    #evaluate
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    scores_table.append(scores[1]*100)
    
    if(scores[1]*100 > bestModel.accuracy):
        bestModel = MyParams(chosen_lr, chosen_neurons, chosen_layers, scores[1]*100, scores[0]*100)
    del model



elapsed_time = time.time() - start_time

print("Czas jaki minal na ", ilosc_iteracji_random_search, " iteracji: ", elapsed_time)

#wypisanie najlepszego modelu
print("To wybralismy: ", bestModel.accuracy, bestModel.lr, bestModel.input_n_n, bestModel.layers_num)
print('\n\n')

# TODO : rysowac wykres najlepszego modelu

# draw histogram ! 

#scores_table_new = np.hstack((scores_table))

#print(scores_table)
#print(scores_table_new)

#plt.hist(scores_table_new, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram wynikow 60 iteracji random search")
#plt.show()

#scores_table_new_2 = np.hstack((scores_table))

#plt.hist(scores_table_new_2, bins=12)  # arguments are passed to np.histogram
#plt.title("Histogram wynikow 60 iteracji random search")
#plt.show()
