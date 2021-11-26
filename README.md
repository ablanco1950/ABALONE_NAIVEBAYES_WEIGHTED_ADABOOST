ABALONE_NAIVEBAYES_WEIGHTED_ADABOOST: Two procedures are attached that use the Abalone file as test and training (https://archive.ics.uci.edu/ml/datasets/abalone). Both start from a treatment of the training part calculating the frequencies corresponding to each value of each field and applying a Naive Bayes probability calculation. In a second step, one of the procedures takes advantage of the previous result to apply weights based on each field to the wrong or true records. The other procedure uses Adaboost, using the adaboost routine published at https://github.com/jaimeps/adaboost-implementation (Jaime Pastor).

A hit rate of around 58% is obtained, that is, in the low range of the existing procedures to treat this multiclass file, which are detailed in the documentation to download from https://archive.ics.uci.edu/ml/datasets/abalone


Resources: Spyder 4

On the c: drive there should be the abalone-1.data file downloaded from https://archive.ics.uci.edu/ml/datasets/abalone

Functioning:

1. NaiveBayes-> Weighted procedure

From Spyder run:

PrepareAbaloneWeghtedNaiveBayes_0.py

Which creates the file C: \ AbaloneWeighted_1.txt that will be used in the later step.

Run from Spyder:

AssignAbaloneWeghtedNaiveBayes.py


2. NaiveBayes-> Adaboost procedure

Run from Spyder:

AbaloneNaiveBayes_Adaboost.py

The console shows how the rate of failures and errors evolves with adaboost, at the end it indicates in which loop the maximum hit rate has occurred that would allow adjusting the number of loops and the smallest number of records detected as erroneous in the log file. Exit:
 C: \ abalone-1Corrected.txt


Cite this software as:

** Alfonso Blanco García ** ABALONE_NAIVEBAYES_WEIGHTED_ADABOOST


References:

https://archive.ics.uci.edu/ml/datasets/abalone

Implementation of AdaBoost classifier

https://github.com/jaimeps/adaboost-implementation
