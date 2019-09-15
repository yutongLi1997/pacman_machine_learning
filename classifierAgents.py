# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
import math

# This is a class for neural network. The network trained a single piece of data a time.
# The first step of this net is compute the total input layer by layer
# and then go back to update the weights
class NeuralNetwork:
    
    def __init__(self, num_inputs, num_hidden, num_outputs): 
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        self.ih_w = [] # a list of lists of the weight from input layer to hidden layer
        self.ho_w = [] # a list of lists of the weight from hidden layer to output layer
        
        # initialize the bias
        self.h_bias = random.random() 
        self.o_bias = random.random() 
        
        # initialize the weight
        self.ih_w = self.initialize_weight(self.num_inputs,self.num_hidden)
        self.ho_w = self.initialize_weight(self.num_hidden,self.num_outputs)
        
    # initialize the weight
    def initialize_weight(self,n_in,n_ou): # Use the number of neurons in layer A and B to compute the weight from layer A to B
        weight_list = []
        for i in range(n_in):
            tmp_list = []
            for j in range(n_ou):
                tmp_list.append(random.random())
            weight_list.append(tmp_list)
        return weight_list
    
    # Calculate the total input of each neurons in hidden layer and output layer
    def total_input(self,inputs,w,bias):
        num_in = len(inputs)
        num_ou = len(w[0])
        computed_input = []
        for m in range(num_ou):
            tmp = 0
            for i in range(num_in):
                # every input times weight
                tmp += inputs[i]*w[i][m]
            tmp += bias
            computed_input.append(tmp)
        return computed_input
    
    # Calculate the output of each neurons in hidden layer and output layer
    def logistic(self,inputs):
        output_list = []
        for i in inputs:
            tmp = 1/(1+math.exp(-i))
            output_list.append(tmp)
        return output_list
    
    # Calculate the total error 
    def e_total(self,target,output):
        error = 0
        for i in range(len(target)):
            error += 0.5 * (target[i]-output[i])**2
        return error
    
    # consider how much the weight from hidden layer to output layer affects the total error, for a single weight
    # d(E_total)/d(wi) = d(E_total)/d(o_output[i])*d(o_output[i])/d(o_input[i])* d(o_input[i])/d(wi)
    # d(E_total)/d(o_output[i]) = -(target[i]-o_output[i])
    def derivative1(self,target,output):
        return -(target-output)
    
    # d(o_output[i])/d(o_input[i]) = o_output[i]*(1-o_output[i])
    def derivative2(self,output):
        return output*(1-output)
    
    # d(o_input[i])/d(wi) = h_output 
    def output_delta(self,target,o_output,h_output):
        v1 = self.derivative1(target,o_output)
        v2 = self.derivative2(o_output)
        return v1*v2*h_output
    
    # consider how much the weight from input layer to hidden layer affects the total error, for a single weight
    # d(E_total)/d(wi) = d(E_total)/d(h_output[i])*d(h_output[i])/d(h_input[i])* d(h_input[i])/d(wi)
    # d(E_total)/d(h_output) = sum(d(E_o_output[i])/d(h_output[i]))
    # d(E_o_output[i])/d(h_output[i]) = d(E_o_output[i])/d(o_input[i])*d(o_input[i])/d(h_output[i])
    # d(E_o_output[i])/d(o_input[i]) = d(E_o_outout[i])/d(o_output[i])*d(o_output[i])/d(o_input[i])
    # d(E_o_output[i])/d(o_input[i]) = d(E_total)/d(o_output[i]) = derivative1()
    # d(o_input[i])/d(h_output[i]) = wi
    # d(h_output[i])/d(h_input[i]) = h_output[i]
    # d(h_input[i])/d(wi) = i_output 
    def hidden_delta(self,inputs,h_output,o_output,target,w):
        v = 0
        for i in range(len(o_output)):
            v1 = self.derivative1(target[i],o_output[i])
            v2 = self.derivative2(o_output[i])
            v3 = v1*v2*w[i] # d(E_o_output[i])/d(h_output[i])
            v += v3
        v4 = self.derivative2(h_output)
        return v*v4*inputs
 
    def train(self, training_inputs,training_outputs):
        # Define a learning rate
        LEARNING_RATE = 0.5
        
        # preprocess the data, transform the output data from a single value to a list of values of 0 or 1. 
        # For example, if the original value is 3, the transformed value is [0,0,0,1]
        def give_values(i,train_list):
            train_list[i] = 1
        output_train = np.zeros(self.num_inputs)
        give_values(training_outputs,output_train)
        training_outputs = output_train
        
        ih_w = self.ih_w
        ho_w = self.ho_w
        # Step 1: feed forward
        # hidden layer
        h_total_input = self.total_input(training_inputs,ih_w,self.h_bias)
        h_output = self.logistic(h_total_input)
        
        # output layer
        o_total_input = self.total_input(h_output,ho_w,self.o_bias)
        o_output = self.logistic(o_total_input)
      
        # Step 2: back propogation
        
        # 1.output layer deltas: d(e_total)/d(wi)
        ho_w_new = []
        for i in range(len(ho_w)):
            tmp_list = []
            for j in range(len(self.ho_w[0])):
                tmp = self.output_delta(training_outputs[j],o_output[j],h_output[i])
                tmp_list.append(tmp)
            ho_w_new.append(tmp_list)
        
        # 2. hidden layer deltas
        ih_w_new = []
        for i in range(len(ih_w)):
            tmp_list = []
            for j in range(len(ih_w[0])):
                tmp = self.hidden_delta(training_inputs[i],h_output[j],o_output,training_outputs,ho_w[j])
                tmp_list.append(tmp)
            ih_w_new.append(tmp_list)
            
        # 3.update the hidden layer to output layer weight
        for i in range(len(ho_w)):
            for j in range(len(ho_w[0])):
                ho_w[i][j] -= ho_w_new[i][j]*LEARNING_RATE
        self.ho_w = ho_w
        
        # 4. Update the input layer to hidden layer weight
        for i in range(len(ih_w)):
            for j in range(len(ih_w[0])):
                ih_w[i][j] -= ih_w_new[i][j]*LEARNING_RATE
        self.ih_w = ih_w
    # predict function
    def predict(self,features):
        # calculate the output of hidden layer
        h_total_input = self.total_input(features,self.ih_w,self.h_bias)
        h_output = self.logistic(h_total_input)
        # calculate the output of output layer
        o_total_input = self.total_input(h_output,self.ho_w,self.o_bias)
        o_output = self.logistic(o_total_input)
        return o_output

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # You may wish to create your classifier here.
        #
        # *********************************************
        
        # train a neural network with 5 hidden neurons and 4 output neurons
        nn = NeuralNetwork(len(self.data[0]), 5, 4)
        for i in range(10000):
            m = i%120
            nn.train(self.data[m], self.target[m])
        self.nn = nn

    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        #
        # Here you should insert code to call the classifier to
        # decide what to do based on features and use it to decide
        # what action to take.
        #
        # *******************************************************
        
        nn = self.nn
        
        # use the trained neural network to predict
        y_pred = nn.predict(features)
        
        # transform the output value from a list of 0 and 1 to a single value
        n = 0
        for i in range(len(y_pred)):
            if y_pred[i]>y_pred[n]:
                n = i
                
        # Get the actions we can try.
        legal = api.legalActions(state)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(self.convertNumberToMove(n), legal)
        #return api.makeMove(Directions.STOP, legal)

