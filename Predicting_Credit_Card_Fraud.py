
# coding: utf-8

# # Predicting Credit Card Fraud

# The goal for this analysis is to predict credit card fraud in the transactional data. I will be using tensorflow to build the predictive model, however, this is my first personal project with tensorflow, so I doubt that I am using it in an overly sophisticated way. If you have any advice/feedback on how to improve my neural network, that would be most welcomed. Thank you!

# In[ ]:

import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


# In[ ]:

df = pd.read_csv("creditcard.csv")


# In[ ]:

df.head()


# The data is mostly transformed from its original form, for confidentiality reasons.

# In[ ]:

df.describe()


# In[ ]:

df.isnull().sum()


# Let's see how time compares across fradulent and normal transactions.

# In[ ]:

print ("Fraud")
print (df.Time[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Time[df.Class == 0].describe())


# In[ ]:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

ax1.hist(df.Time[df.Class == 1])
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0])
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# The 'Time' feature looks pretty similar across both types of transactions.

# Perhaps the amount is different between the two types.

# In[ ]:

print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())


# In[ ]:

plt.figure(figsize=(10,4))

plt.subplot(211)
plt.hist(df.Amount[df.Class == 1])
plt.yscale('log')
plt.title('Fraud')
plt.grid(True)

plt.subplot(212)
plt.hist(df.Amount[df.Class == 0])
plt.yscale('log')
plt.title('Normal')
plt.grid(True)

plt.xlabel("Amount")
plt.show()


# Amount also looks very similar across both types of transactions. Let's try looking at these two features together.

# In[ ]:

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# Nothing useful here. I was hoping that we would be able to create a new feature or alter at least one of these features to help improve the model...I'm afraid that won't happen.

# In[ ]:

#Create a new feature for normal (non-fraudulent) transactions.
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0


# In[ ]:

df = df.rename(columns={'Class': 'Fraud'})


# In[ ]:

#492 fraudulent transactions, 284,315 normal transactions.
#0.172% of transactions were fraud. 
print(df.Normal.value_counts())
print()
print(df.Fraud.value_counts())


# In[ ]:

df.head()


# In[ ]:

#Create dataframes of only Fraud and Normal transactions
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]


# In[ ]:

#Set X_train equal to half of the fraudulent transactions
X_train = Fraud[::2]
count_Frauds = len(X_train)

#Add 50,000 normal transactions to X_train
X_train = pd.concat([X_train, Normal.sample(n = 50000)], axis = 0)

#X_test contains all the transaction not in X_train
X_test = df.loc[~df.index.isin(X_train.index)]


# In[ ]:

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)


# In[ ]:

y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Normal], axis=1)


# In[ ]:

X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)


# In[ ]:

#Check to ensure all of the training/testing dataframes are of the correct length
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


# In[ ]:

'''
ratio will act as an equal weighting system for our model. Because we want to reduce the mean squared error between
our predicted and target values, we want the initial sums of our labels to be equal. ratio will multiply the Fraud 
values so that they equal the sum of the Normal values (each Normal value equals 1).
'''
ratio = len(X_train)/count_Frauds 

y_train.Fraud *= ratio
y_test.Fraud *= ratio


# In[ ]:

#Names of all of the features in X_train
features = X_train.columns.values

#Transform each feature so that it has a mean of 0 and standard deviation of 1; this helps with training the model.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std


# ## Train the Neural Net

# In[ ]:

inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_test = X_test.as_matrix()
inputY_test = y_test.as_matrix()


# In[ ]:

#Multiplier maintains a fixed ratio of nodes between each layer
mulitplier = 1.5 

#Number of nodes in hidden layer 1
hidden_nodes1 = 15
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)

#input
x = tf.placeholder(tf.float32, [None, 30]) #there are 30 inputs

#layer 1
W1 = tf.Variable(tf.zeros([30, hidden_nodes1]))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

#layer 2
W2 = tf.Variable(tf.zeros([hidden_nodes1, hidden_nodes2]))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

#layer 3
W3 = tf.Variable(tf.zeros([hidden_nodes2, hidden_nodes3])) 
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

#layer 4
W4 = tf.Variable(tf.zeros([hidden_nodes3, 2])) 
b4 = tf.Variable(tf.zeros([2]))
y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

#output
y = y4
y_ = tf.placeholder(tf.float32, [None, 2])


# In[ ]:

#Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)

#We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Correct prediction if the most likely value (Fraud or Normal) from softmax equal the target value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:

#Initialize variables and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# In[ ]:

#Parameters
learning_rate = 0.0005
training_epochs = 30000
display_step = 100
n_samples = y_train.size
accuracy_summary = [] #Record accuracy values for plot
cost_summary = [] #Record cost values for plot

for i in range(training_epochs):  
    sess.run([optimizer], feed_dict={x: inputX, y_: inputY})
    
    # Display logs per epoch step
    if (i) % display_step == 0:
        train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX, y_: inputY})
        print ("Training step:", i,
               "Accuracy =", "{:.5f}".format(train_accuracy), 
               "Cost = ", "{:.5f}".format(newCost))
        accuracy_summary.append(train_accuracy)
        cost_summary.append(newCost)
        
print()
print ("Optimization Finished!")
training_accuracy = sess.run(accuracy, feed_dict={x: inputX, y_: inputY})
print ("Training Accuracy=", training_accuracy)
print()
testing_accuracy = sess.run(accuracy, feed_dict={x: inputX_test, y_: inputY_test})
print ("Testing Accuracy=", testing_accuracy)


# In[ ]:

#Use to find testing accuracy if I interrupt the training before it is finished
predicted = tf.argmax(y, 1)
testing_accuracy, testing_predictions = sess.run([accuracy,predicted], feed_dict={x: inputX_test, y_: inputY_test })
print(testing_accuracy)


# In[ ]:

#Plot accuracy and cost summary

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

ax1.plot(accuracy_summary)
ax1.set_title('Accuracy')

ax2.plot(cost_summary)
ax2.set_title('Cost')

plt.xlabel('Epochs (x100)')
plt.show()


# In[ ]:

confusion_matrix(y_test.Normal, testing_predictions)


# To summarize the confusion matrix: 
# 
# Correct Fraud: 203
# 
# Incorrect Fraud: 43
# 
# Correct Normal: 233,671
# 
# Incorrect Normal: 644

# # Summary

# Although the neural network can detect most of the fraudulent transactions (82.5%), there are still some that got away. About 0.27% of normal transactions were classified as fraudulent, which can unfortunately add up very quickly given the large number of credit card transactions that occur each minute/hour/day. Nonetheless, I hope that you have learned something about tensorflow from this analysis and in you have any advice on how to improve the model, I would really appreciate that. Thank you!

# In[ ]:



