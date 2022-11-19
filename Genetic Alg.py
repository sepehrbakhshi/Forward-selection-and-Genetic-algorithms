# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:37:23 2019

@author: Sepehr
"""
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys
import matplotlib.pyplot as plt

class GeneticAlgorithm(object):

      def __init__(self,p_size,m_prob,n_generations,c_rate , x_train,x_test
                   ,y_train, y_test , cost, max_iter):
        self.population_size = p_size
        self.mutation_rate = m_prob
        self.population = []
        self.chromose_len = 21
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.cost_per_feature = cost
        self.fitness_result = []
        self.max_iteration = max_iter
        self.deleting_rate = 0
        self.list_of_fitnesses = []
        for i in range(0 , self.population_size):
            one_rate = random.random()
            if (one_rate < 0.05):
                while(one_rate < 0.05):
                    one_rate = random.random()
            chromosome = np.zeros(shape=(21,1))
            indices = np.random.choice(np.arange(self.chromose_len), replace=False,
                           size=int(self.chromose_len * one_rate))
            chromosome[indices] = 1
            self.population.append(chromosome.T)
        self.population = np.array(self.population)
        print(self.population)


      def fitness(self):
          #svclassifier = SVC(kernel='linear')

          #svclassifier = SVC(kernel='poly', degree=20)
          #mlp = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=700)
          #svclassifier = SVC(kernel='poly', degree=3)
          classifier = DecisionTreeClassifier()

          self.fitness_result = []
          for i in range( 0 , self.population_size ):

              index = np.where(self.population[i] == 1)
              index = np.array(index)
              idx_IN_columns = index[1]

              chrom_train_x = self.x_train[(slice(None) , idx_IN_columns)]
              chrom_test_x = self.x_test[(slice(None) , idx_IN_columns)]
              #chrom_train_x = chrom_train_x.reshape(10274, chrom_train_x.shape[1])

              #chrom_test_x = chrom_test_x.reshape(9503, chrom_test_x.shape[1])
              """
              mlp.fit(chrom_train_x, self.y_train)
              y_pred = mlp.predict(chrom_test_x)

              """
              """
              svclassifier.fit(x_train_oversampled, labels_train_oversampled)

              y_pred = svclassifier.predict(x_test_oversampled)

              """
              classifier.fit(chrom_train_x, self.y_train)
              y_pred = classifier.predict(chrom_test_x)

              acc = accuracy_score(self.y_test, y_pred)

              y_pred_train = classifier.predict(chrom_train_x)
              acc_train = accuracy_score(self.y_train, y_pred_train)
              cost_chrom = self.population[i][0,0:20]

              index = np.where(cost_chrom == 1)
              index = np.array(index)
              idx_IN_columns = index[0]

              pop_cost = self.cost_per_feature[(idx_IN_columns)]
              print(pop_cost)
              print(self.population)
              pop_cost = np.sum(pop_cost)

              if(self.population[i][0,20] == 1 and self.population[i][0,18] == 0 and self.population[i][0,19] == 0):
                  pop_cost = pop_cost + self.cost_per_feature[18] + self.cost_per_feature[19]
              elif(self.population[i][0,20] == 1 and self.population[i][0,18] == 1 and self.population[i][0,19] == 0):
                  pop_cost = pop_cost +  self.cost_per_feature[19]
              elif(self.population[i][0,20] == 1 and self.population[i][0,18] == 0 and self.population[i][0,19] == 1):
                  pop_cost = pop_cost +  self.cost_per_feature[18]
              print("Acc and cost : ")
              print(acc)
              print("acc train")
              print(acc_train)
              print("++")
              print(pop_cost)

              pop_cost = pop_cost/76.11

              c = 1 - pop_cost

              w_1 = 0.7
              w_2 = 0.3

              #pop_cost = pop_cost/76.11
              #fitness = (acc * w_1)+ (w_2*c)
              fitness = (acc - (pop_cost/ (acc + 1))) + 76.11
              print("Normalized cost : ")
              print(pop_cost)
              self.fitness_result.append(fitness)
              print("Fitness :")
              print(fitness)

              print("-------------------------")

      def crossover(self):
           pop = []
           pop = self.population
           last_fitness_results = self.fitness_result
           last_fitness_results = np.array(last_fitness_results)
           self.list_of_fitnesses.append(last_fitness_results)
           self.deleting_rate = self.population_size * 0.4
           n = 0
           while(n < self.deleting_rate):
               index = np.where(last_fitness_results == last_fitness_results.min())

               index = index[0]
               index = np.array(index)
               if(len(index) > 1):
                   while(len(index) != 1):
                       index = np.delete(index, 0,0)

               last_fitness_results = np.delete(last_fitness_results, index,0)
               pop = np.delete(pop, index,0)
               n += 1
           if(pop.shape[0] < 6):
               sys.exit()
           self.population = []

           new_pop = []

           print("new population :")
           print(pop)
           print("-------------------------")
           np.random.shuffle(pop)
           counter = 1
           for i in range(0 , (int)(self.deleting_rate/4)):
               chrom_1 = []
               chrom_2 = []
               chrom_1 = pop[i].copy()
               chrom_2 = pop[i+1].copy()

               cross_over_rate = 0.8

               cross_index = (int)(pop[i].shape[1]*cross_over_rate)

               print("Cross_index : ")
               print(cross_index)
               j = 0

               for j in range(0,cross_index):
                   if(chrom_1[0,j] == 0 and chrom_2[0,j] == 1 ):
                       chrom_1[0,j] = 1
                       chrom_2[0,j] = 0
                   elif(chrom_1[0,j] == 1 and chrom_2[0,j] == 0 ):
                       chrom_1[0,j] = 0
                       chrom_2[0,j] = 1

               one_bit = random.random()
               one_bit = (int)(one_bit * 16)

               tmp = 0
               number = chrom_1[0,one_bit]
               print(number )
               if(number == 0):
                   tmp  = 1
               if(number == 1):
                   tmp = 0
               chrom_1[0,one_bit] = tmp
               one_bit_2 = random.random()
               one_bit_2 = (int)(one_bit_2 * 16)
               tmp = 0
               number = 0
               number = chrom_2[0,one_bit_2]
               if(number == 0):
                   tmp =  1
               if(number == 1):
                   tmp = 0

               chrom_2[0,one_bit_2] = tmp

               new_pop.append(chrom_1)

               new_pop.append(chrom_2)

               counter += 1
           #print("Counter is finished")

           #self.population = []

           #np.random.shuffle(pop)

           for i in range(counter , (counter + (int)(self.deleting_rate/2))):
              chrom_1 = []
              chrom_1 = pop[i].copy()
              mutation_rate = 0.3
              #mutation_rate = random.random()

              indices = np.random.choice(np.arange(self.chromose_len), replace=False,
                              size=int(self.chromose_len * mutation_rate))
              #print("Before Mutation :")
              #print(self.population[i])

              print(indices)

              for j in range(0 , indices.shape[0]):
                  mute_index = indices[j]
                  #print(chrom_1.shape)
                  if(chrom_1[0,mute_index] == 1):

                      chrom_1[0,mute_index]  = 0
                  elif(chrom_1[0,mute_index]  == 0):

                      chrom_1[0,mute_index]  = 1
              new_pop.append(chrom_1)
              print(counter)
              counter += 1

           #print(pop.shape)
           #new_pop.append(pop)
           new_pop = np.array(new_pop)

           print(new_pop.shape)
           #new_pop = np.array(new_pop)
           #new_pop.tolist()

           #new_pop = np.array(new_pop)

           print("New Pop after crossover and mutation: ")
           #self.population = np.array(self.population)
           #self.population.tolist()
           self.population = np.vstack((pop,new_pop))
           #self.population = new_pop
           for i in range(0 , self.population_size ):
               if(np.count_nonzero(self.population[i]) == 0 ):
                   self.population[i][0,4] = 1
           self.population.reshape(self.population_size, 1, 21)
           print(self.population)
                #print(indices)
              #print("After Mutation :")
              #print(self.population[i])
      def train(self):
          for i in range(0 , self.max_iteration):
              self.fitness()
              self.crossover()


#####################################################

trainData_ = np.loadtxt(fname="train_data.txt")
testData = np.loadtxt(fname="test_data.txt")

def sampling(data,factor):
    ones_twos = np.zeros((0,22))
    for d in data:
        if (d[21] == 2 or d[21] == 1):
            ones_twos = np.vstack([ones_twos,d])

    newData = data
    for _ in range(factor):
        newData = np.vstack([newData,ones_twos])

    random.shuffle(newData)
    return newData

oversamplingFactor = 22
trainData = sampling(trainData_, oversamplingFactor)
testData = sampling(testData, oversamplingFactor)



X_train = trainData[:, :-1]
y_train = trainData[:, -1]
X_test = testData[:, :-1]
y_test = testData[:, -1]
counter_1 = 0
counter_2 = 0
counter_3 = 0
for i in range(0,y_test.shape[0] ):
    if(y_train[i] == 1):
        counter_1 += 1
    if(y_train[i] == 2):
        counter_2 += 1
    if(y_train[i] == 3):
        counter_3 += 1

loadedData_cost = pd.read_csv('cost.txt', delimiter= '\s+', header=None)
#loadedData_cost = pd.read_csv('cost.txt', delimiter = )
loadedData_cost = np.array(loadedData_cost)
loadedData_cost = loadedData_cost[(slice(None),1)]

max_iter = 80
ga = GeneticAlgorithm(10 , 4 ,10 ,7 ,X_train,X_test,
                      y_train,y_test ,loadedData_cost,max_iter)
ga.train()
last_fit = np.array(ga.list_of_fitnesses)
average_of_fitnesses = []
max_fitnesses = []
for i in range( 0 , 80):
    average_of_fitnesses.append(np.sum(last_fit[i])/10)
for i in range( 0 , 80):
    max_fitnesses.append(np.max(last_fit[i]))

print(last_fit)
print(last_fit.shape)
print("-----------------------------------------------------")
print(average_of_fitnesses)
print("-----------------------------------------------------")

ga.fitness()
y = []
for i in range(1,81):
    y.append(i)
y = np.array(y)
#max_fitnesses, y = zip(*sorted(zip(average_of_fitnesses, y)))

plt.plot(y,average_of_fitnesses,label = "Average Fitness Function")
txt = "Average fitness function per iteration"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/test1.png')
plt.show()

plt.plot(y,max_fitnesses,label = "Maximum Fitness Function")
txt = "Maximum fitness function per iteration"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('C:/Users/Sepehr/Documents/test2.png')
plt.show()
print(counter_1)
print(counter_2)
print(counter_3)


first_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 1):
        first_class.append(testData[i])
first_class = np.array(first_class)
first_class_x = first_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = first_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("first")
print(acc)
second_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 2):
        second_class.append(testData[i])
second_class = np.array(second_class)
first_class_x = second_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = second_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)
third_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 3):
        third_class.append(testData[i])
third_class = np.array(third_class)
first_class_x = third_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = third_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)


############
print("$$$$$$$$$$$$")
first_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 1):
        first_class.append(testData[i])
first_class = np.array(first_class)
first_class_x = first_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = first_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("first")
print(acc)
second_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 2):
        second_class.append(testData[i])
second_class = np.array(second_class)
first_class_x = second_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = second_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)
third_class = []
for i in range(0, testData.shape[0]):
    if (testData[i,21] == 3):
        third_class.append(testData[i])
third_class = np.array(third_class)
first_class_x = third_class[: , 0:21]
first_class_x = np.array(first_class_x)
print(first_class_x.shape)
first_class_y = third_class[: , 21]
first_class_y = np.array(first_class_y)
y_pred = model.predict( first_class_x[:,f])
acc = accuracy_score(first_class_y, y_pred)
print("second")
print(acc)

"""
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
"""
print(counter_1)
print(counter_2)
print(counter_3)
