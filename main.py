#####Find several readability or textual difficulty formulas/algorithms and obtain the corresponding score for a given text
###Probably we should take into consideration the different factors of readability: formulas, text,...

import pickle

from readability_score.calculators.ari import *
from readability_score.calculators.colemanliau import *
from readability_score.calculators.dalechall import *
from readability_score.calculators.flesch import *
from readability_score.calculators.fleschkincaid import *
from readability_score.calculators.smog import *
import os, os.path

import matplotlib.pyplot as plt
import numpy as np
#print numpy.__file__
from sklearn import linear_model
from scipy.optimize import minimize

def bundle(file):
    fk = FleschKincaid(open(file).read(),locale='./hyph_en_US.dic')
    dc = DaleChall(open(file).read(),simplewordlist=word_list, locale='./hyph_en_US.dic')
    ari = ARI(open(file ).read(), locale='./hyph_en_US.dic')
    cl = ColemanLiau(open( file ).read(), locale='./hyph_en_US.dic')
    sm = SMOG(open( file ).read(), locale='./hyph_en_US.dic')
    #Add more formulas that account for the readability of formulas present in the text and the number of figures

    return fk.us_grade, dc.us_grade, ari.us_grade, cl.us_grade, sm.us_grade

def word_count(file):
    fk = FleschKincaid(open(file).read(),locale='./hyph_en_US.dic')

    return fk.scores['word_count']

def cost_fun(x):
    """The cost function using ordinary least square"""
    result=0
    for i in range(file_num):
        if i<(file_num/3):
            fk_sc, dc_sc, ari_sc, cl_sc, sm_sc = bundle('./texts/text_eas'+str(i+1)+'.txt')
            result+=((fk_sc*x[0]+dc_sc*x[1]+ari_sc*x[2]+cl_sc*x[3]+sm_sc*x[4])/5-6)**2
            #print 'easy', i
            #print result
        elif i<(2*file_num/3):
            fk_sc, dc_sc, ari_sc, cl_sc, sm_sc = bundle('./texts/text_norm'+str(i-2)+'.txt')
            result+=((fk_sc*x[0]+dc_sc*x[1]+ari_sc*x[2]+cl_sc*x[3]+sm_sc*x[4])/5-10)**2
            #print 'norm', i
            #print result
        else:
            fk_sc, dc_sc, ari_sc, cl_sc, sm_sc = bundle('./texts/text_dif'+str(i-5)+'.txt')
            result+=((fk_sc*x[0]+dc_sc*x[1]+ari_sc*x[2]+cl_sc*x[3]+sm_sc*x[4])/5-13)**2
            #print 'dif', i
            #print result
    return result

def final_formula(weights,filename):
    fk_sc, dc_sc, ari_sc, cl_sc, sm_sc = bundle(filename)
    return (fk_sc*weights[0]+dc_sc*weights[1]+ari_sc*weights[2]+cl_sc*weights[3]+sm_sc*weights[4])/5

####Normalize the given score so that all different algorithms have the same max and min. The normalization corresponds to the grade

file_num=9
#Load the list of easy_words for Dale Chall
word_list=pickle.load( open( "dale_chall.p", "rb" ) )


##Now we proceed to add the results together to obtain a more meaningful result
##We are going to give weights to each formula, build a cost function and minimize it.
##We have obtained 9 texts and assigned the grade score: 3 easy (6), 3 normal (10) and 3 difficult (13).
#Increase the number of files and the corresponding score in order to obtain better results.
#The weight could be improved by assigning a caracteristic set of weights to different texts.

##Initial values
x0 = np.array([1, 1, 1, 1, 1])

##Minimize the cost function
print 'Minimizing the cost function...'
print '(It make take some time, check the backup weigths in the code)'
res = minimize(cost_fun, x0, method='nelder-mead', options={'maxiter':2, 'disp': True})

print 'Resulting weights',(res.x)
#As it takes a long time, here are some sample weights
backup_weigths=[ 0.17602172,  0.78345087,  0.30895142,  1.42169477,  1.19454384]

######How your compound score can be operationalized to provide more accurate reading time estimates.
###Here we need to take into consideration the length of the text and the readability.
### So we determine the average time per word and then just multiply.
#We start by determining the final grade_score of the documents to be used in the reading time estimates.

num_word_list=[]
score_list=[]
dif_final_score=final_formula(backup_weigths,'./texts/text_dif_test.txt')
score_list.append(dif_final_score)
num_word_list.append(word_count('./texts/text_dif_test.txt'))
norm_final_score=final_formula(backup_weigths,'./texts/text_norm_test.txt')
score_list.append(norm_final_score)
num_word_list.append(word_count('./texts/text_norm_test.txt'))
easy_final_score=final_formula(backup_weigths,'./texts/text_eas_test.txt')
score_list.append(easy_final_score)
num_word_list.append(word_count('./texts/text_eas_test.txt'))

print '3 scores', dif_final_score, norm_final_score, easy_final_score

#My reading times in seconds
#This value could be measured using the command 'start = timeit.timeit()' for the time the user spends in a given page
#or document. However, we should proceed with caution in the details of the measurement to account for: non-reading time,
#unread text,....

reading_times=[95, 104, 48]


#Reading time per word
reading_per_word=[float(reading_times[i])/num_word_list[i] for i in range(3)]

print 'Reading time per word in seconds', reading_per_word

#Now we perform linear regression with respect to readability and seconds to read a word.
#We could also perform higher level regression or other techniques and compare results

train_Y=reading_per_word
train_X=score_list

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(np.reshape(train_X,(3,1)), np.reshape(train_Y,(3,1)))

test_X=np.reshape([5,10],(2,1))

# Plot outputs
plt.scatter(np.reshape(train_X,(3,1)), np.reshape(train_Y,(3,1)),  color='black')
plt.plot(test_X, regr.predict(test_X), color='blue',linewidth=3)

plt.axis([5, 10, 0, 0.25])
plt.xlabel("Readability score")
plt.ylabel("Seconds per word")

plt.show()

#More complex techniques can be applied for bigger datasets to improve accuracy.
#The basic idea is to have a regression that gives the seconds per word for a given readability document.
#For example for a document with 10 readability score and 500 words, the amount of time that we expect would be:

print 'Number of minutes for the given text', float(regr.predict(10)*500)/60

### This result applies only for texts with words, for formulas we should operate in a similar way with different time scale
#and the same goes for figures.
###e.g.: n_words * t_word_difficulty_text + n_formulas * t_formula_difficulty + n_figures * t_figure_difficulty....
## This proccess should be repeated for formulas and images.
# Another factor that we have not taken into consideration is the typography of the document.












