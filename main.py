#####Find several readability or textual difficulty formulas/algorithms and obtain the corresponding score for a given text
###Probably we should take into consideration the different factors of readability: formulas, text,...

from readability_score.calculators.fleschkincaid import *
from readability_score.calculators.dalechall import *
from readability_score.calculators.ari import *
from readability_score.calculators.colemanliau import *
from readability_score.calculators.flesch import *
from readability_score.calculators.smog import *
import pickle
import matplotlib.pyplot as plt
#import numpy as np
from sklearn import datasets, linear_model
from hyphen import Hyphenator, dict_info
from hyphen.dictools import *
import nltk

#nltk.download('punkt')

#print dict_info.keys()

#awordList=['a','the']

sol_list=[]
#No upper bound formula with min -3.4
#Units are grade
fk = FleschKincaid(open('text.txt').read(),locale='./hyph_en_US.dic')
print fk.us_grade, fk.min_age
#print fk.scores
sol_list.append(fk.us_grade)

#Load the list of easy_words for Dale Chall

word_list=pickle.load( open( "dale_chall.p", "rb" ) )

#Units are grade
dc = DaleChall(open( 'text.txt' ).read(),simplewordlist=word_list, locale='./hyph_en_US.dic')

print dc.us_grade, dc.min_age
sol_list.append(dc.us_grade)

ari = ARI(open( 'text.txt' ).read(), locale='./hyph_en_US.dic')

print ari.us_grade, ari.min_age
sol_list.append(ari.us_grade)

cl = ColemanLiau(open( 'text.txt' ).read(), locale='./hyph_en_US.dic')

print cl.us_grade, cl.min_age
sol_list.append(cl.us_grade)


#Probably redundant and difficult to put in common with the others
ff = Flesch(open( 'text.txt' ).read(), locale='./hyph_en_US.dic')

print  ff.reading_ease, ff.scores
print 1-ff.reading_ease/100, ff.reading_ease/100

sm = SMOG(open( 'text.txt' ).read(), locale='./hyph_en_US.dic')

print sm.us_grade, sm.min_age
sol_list.append(sm.us_grade)

print sum(sol_list)/len(sol_list)


train_Y=sol_list
train_X=[0,1,2,3,4,5]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_X, train_Y)

test_X=[2.5]

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
#print("Residual sum of squares: %.2f"
#      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
#plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(test_X, regr.predict(test_X), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

####Normalize the given score so that all different algorithms have the same max and min (is any other normalization required?)






#####Add the scores in a non-trivial way, netflix example, most probably machine learning required to determine the weights.





######How your compound score can be operationalized to provide more accurate reading time estimates.
###Here we need to take into consideration the length of the text and the readability.
### So we determine the average time per word (or equivalent minimum value) and then just multiply.
### This applies only for texts with words, for formulas we should operate in a similar way with different time scale.
###e.g.: n_words * t_word_difficulty_text + n_formulas * t_formula_difficulty






