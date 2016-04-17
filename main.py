#####Find several readability or textual difficulty formulas/algorithms and obtain the corresponding score for a given text
###Probably we should take into consideration the different factors of readability: formulas, text,...



####Normalize the given score so that all different algorithms have the same max and min (is any other normalization required?)





#####Add the scores in a non-trivial way, netflix example, most probably machine learning required to determine the weights.





######How your compound score can be operationalized to provide more accurate reading time estimates.
###Here we need to take into consideration the length of the text and the readability.
### So we determine the average time per word (or equivalent minimum value) and then just multiply.
### This applies only for texts with words, for formulas we should operate in a similar way with different time scale.
###e.g.: n_words * t_word_difficulty_text + n_formulas * t_formula_difficulty






