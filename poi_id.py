#!/usr/bin/python

import sys
import pickle
sys.path.append('../tools/')

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#numpy
import numpy

# Gaussian Naive Bayes, SGD, kNN and DecisionTree classifiers
# will be evaluated and the best performing one will be tuned
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#cross validation is used to evaluate the recall of classifiers in a 10-fold fashion
from sklearn.model_selection import cross_val_score

# f1_weighted is used in replacement of conventional f1 since the dataset is unbalanced

METRIC_FOR_TUNING = 'f1_weighted' #'f1' #'f1_weighted' #'accuracy' #'recall'
FOLDS_FOR_TUNING = 10
N_THREADS_FOR_TUNING = 2

# gridsearchcv is used since all folds are stratified using StratifiedKFold internally
from sklearn.model_selection import GridSearchCV


########################
# BASIC INFO GATHERING #
########################

def getBasicInfo(my_dataset):
    totalInstances = 0
    numberOfNaNs = 0
    numberPOI = 0
    datasetFeatures = []
    for name, features in my_dataset.iteritems():
        totalInstances += 1
        for f in features:
            if f not in datasetFeatures and f != 'poi' \
                    and f != 'email_address':
                datasetFeatures.append(f)
            if features[f] == 'NaN':
                numberOfNaNs += 1
            if f == 'poi' and features[f] == True:
                numberPOI += 1
    return (totalInstances, numberPOI, len(features))


##########
# TUNING #
##########

def tuneClassifier(classifier, X, Y):
    if type(classifier) is SGDClassifier:
        return tuneSGD(X, Y)
    if type(classifier) is GaussianNB:
        return tuneGaussianNB(X, Y)
    if type(classifier) is KNeighborsClassifier:
        return tuneKNN(X, Y)
    if type(classifier) is DecisionTreeClassifier:
        return tuneDecisionTree(X, Y)

def tuneKNN(X, Y):
    # values from k ranging 1 to 20
    #manhattan, euclidian and l_3 norm
    kValues = [i for i in range(1, 21)] #[1, 3, 5, 7, 9]
    parameters = {'n_neighbors': kValues, 'p': [1, 2, 3], 'algorithm': ['brute']}
    grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters,
                        n_jobs=N_THREADS_FOR_TUNING, cv=FOLDS_FOR_TUNING,
                        scoring=METRIC_FOR_TUNING)
    grid.fit(X, Y)
    bestEstimator = grid.best_estimator_
    bestScore = grid.best_score_
    return (bestEstimator, bestScore)

def tuneSGD(X,Y):
    parameters = {'loss': ['hinge', 'log', 'squared_hinge', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet']}
    grid = GridSearchCV(estimator=SGDClassifier(), param_grid=parameters,
                        n_jobs = N_THREADS_FOR_TUNING, cv=FOLDS_FOR_TUNING,
                        scoring=METRIC_FOR_TUNING)
    grid.fit(X, Y)
    bestEstimator = grid.best_estimator_
    bestScore = grid.best_score_
    return (bestEstimator, bestScore)

def tuneGaussianNB(X, Y):
    # doesn't appear to have any parameters to be tuned,
    # and thus, returns a single gaussian nb learner trained using X and Y
    gnb = GaussianNB()
    gnb.fit(X, Y)
    scores = cross_val_score(gnb, X, Y, cv=FOLDS_FOR_TUNING, scoring=METRIC_FOR_TUNING)
    return (gnb, scores.mean())

def tuneDecisionTree(X, Y):
    parameters = {'criterion': ('gini', 'entropy'),
                  'splitter': ('best', 'random')}
    grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters,
                        n_jobs=N_THREADS_FOR_TUNING, cv=FOLDS_FOR_TUNING,
                        scoring=METRIC_FOR_TUNING)
    grid.fit(X, Y)
    bestEstimator = grid.best_estimator_
    bestScore = grid.best_score_
    return (bestEstimator, bestScore)


############################
# CREATION OF NEW FEATURES #
############################

def addFeatures(my_dataset):
    # the new attributes
    newFeatures = ['total_asset', 'fraction_of_messages_with_poi']
    # asset-related variables
    assets = ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
    # message-related variables
    messages = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi']
    for name, features in my_dataset.items():
        validAssetFeature = True
        validMessageFeature = True
        for key in assets:
            if features[key] == 'NaN':
                validAssetFeature = False
                break
        for key in messages:
            if features[key] == 'NaN':
                validMessageFeature = False
                break
        if validAssetFeature:
            # sum the total asset if data are valid
            features[newFeatures[0]] = sum([features[key] for key in assets])
        else:
            # data is invalid, so NaN is set
            features[newFeatures[0]] = 'NaN'


        #### TESTING NOW FOR MESSAGE FEATURE
        if validMessageFeature:
            # computes the ratio between POI messages and all messages
            all_messages = features['to_messages'] + features['from_messages']
            messages_with_poi = features['from_poi_to_this_person'] + features['from_this_person_to_poi']
            if all_messages > 0:
                features[newFeatures[1]] = float(messages_with_poi) / all_messages
            else:
                features[newFeatures[1]] = 'NaN'
        else:
            # data is invalid, so NaN is set
            features[newFeatures[1]] = 'NaN'
    return (my_dataset, newFeatures)

#####################
# FEATURE SELECTION #
#####################

# I did here a wrapper-like approach, where k features are incrementally selected given their chi^2 score,
# and the best result for the evaluated metric in k \in [1;10] is picked for a given classifier
def selectAttributes(X, Y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectFpr
    # stores the best K, Score, the features selected as a list of strings
    # and the reduced dimensionality problem
    bestScoreProps = (0, 0.0, [], None)
    for i in range(1, len(X[0])):
        propsSelection = SelectKBest(k=i).fit(X, Y)
        scoresAfterSelection = propsSelection.scores_
        indicesOfSelectedAttributes = numpy.argpartition(scoresAfterSelection, i)[-i:]
        Xselected = SelectKBest(k=i).fit_transform(X, Y)
        score = cross_val_score(GaussianNB(), Xselected, Y, cv=FOLDS_FOR_TUNING, scoring=METRIC_FOR_TUNING)
        _, bestScore, _, _ = bestScoreProps
        avgScore = numpy.average(score)
        if avgScore > bestScore:
            bestScoreProps = (i, avgScore, indicesOfSelectedAttributes, Xselected)

    bestK, bestScore, selectedFeatures, Xnew = bestScoreProps
    print '\t Best result was n=%d with a %.3f score and the selected features were ' % (bestK, bestScore), selectedFeatures
    return (selectedFeatures, Xnew)



##################
# DATA CLEANSING #
##################

def cleanData(X):
    # Replaces all NaNs in the dataset with a 0
    # and replaces all negative values into their absolute ones
    for key, line in X.items():
        for feature in line:
            if line[feature] == 'NaN':
                line[feature] = 0.0
            if line[feature] < 0.0:
                line[feature] = abs(line[feature])

##################
# DATA RESCALING #
##################
def rescaleData(X):
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    return X

############
# OUTLIERS #
############

def removeOutliers(dataset):

    # The first thing done is to remove the TOTAL line, which is the sum
    # of all others and is not really a representative example for training
    print '\t Removing TOTAL...'
    dataset.pop('TOTAL', 0)
    print '\t Removing THE TRAVEL AGENCY IN THE PARK...'
    dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0)


    #gets the number of attributes in the dataset with exception of the class 'poi'
    numberFeatures = len(dataset.itervalues().next()) - 1


    #count NaNs per name in the dataset
    nanPerName = {}
    for name, value in dataset.iteritems():
        qtdNan = 0
        for feature, value in value.iteritems():
            if value == 'NaN':
                qtdNan += 1
        nanPerName[name] = qtdNan

    #for k, v in nanPerName.iteritems():
    #    print k, v
    print '\t Removing all the users that have more than 95% of the features with invalid values'
    #removes all the names with more then 95% of attributes with NaNs
    for name, qtd in nanPerName.iteritems():
        if qtd >= 0.95 * numberFeatures:
            print '\t\t Removing', name
            dataset.pop(name, 0)




########
# MAIN #
########

print 'Loading data...'

### Load the dictionary containing the dataset
with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be 'poi'.
features_list = [ #label
                 'poi',
                 # financial attributes (all in US dollars)
                 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 # email attributes (number of email messages, with the exception of email_address, which is a string)
                 'to_messages', #'email_address',
                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Gets some basic info on the dataset and prints it
totalInstances, numberPOI, numberFeatures = getBasicInfo(my_dataset)
print '\t Number of instances = %d' % totalInstances
print '\t Number of POIs = %d' % numberPOI
print '\t Number of features = %d' % numberFeatures
print '\t Ratio of POIs and non POIs = (%.2f, %.2f)' % (float(numberPOI)/totalInstances, float(totalInstances-numberPOI)/totalInstances)


### Task 3: Create new feature(s)
my_dataset, newFeatures = addFeatures(my_dataset)
features_list += newFeatures

#debug printing of the dataset
# for key, value in my_dataset.iteritems():
#     print key, value

# getting the features' names in the sequence they are stored in memory
# for feature in my_dataset.values()[0]:
#     print feature


### Task 2: Remove outliers
print 'Removing outliers...'
removeOutliers(my_dataset)

# Cleans the data set
print 'Cleaning the dataset...'
cleanData(my_dataset)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# line = ['%.2f' % elem for elem in features[0]]
# print line


# Rescale data
print 'Normalizing data...'
features = rescaleData(features)

##### Run feature selection in wrapper-like approach
print 'Performing automatic feature selection...'

indicesFeaturesSelected, features = selectAttributes(features, labels)

keysFeatures = []
for v in my_dataset.values()[0]:
    if v != 'poi':
        keysFeatures.append(v)

#pops all features that were not selected from my_dataset and from features_list
for i in range(0, len(keysFeatures)):
    if i not in indicesFeaturesSelected:
        for _, v in my_dataset.iteritems():
            del v[keysFeatures[i]]

for i in range(0, len(keysFeatures)):
    if i not in indicesFeaturesSelected:
        f = keysFeatures[i]
        if f in features_list:
            features_list.remove(f)

#prints the selected attributes
print '\t\t Final set of attributes = ', features_list


### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

print 'Tuning classifiers...'

classifiers = [('GaussianNB', GaussianNB())]
               #('SGD', SGDClassifier())]
               #('KNN', KNeighborsClassifier())]
               # ('Decision Tree', DecisionTreeClassifier())]
avgScores = []

for name, classifier in classifiers:
    print '\tTuning', name, '...'
    tunedClassifier, tunedScore = tuneClassifier(classifier, features, labels)
    scores = cross_val_score(tunedClassifier, features, labels, cv=FOLDS_FOR_TUNING, scoring=METRIC_FOR_TUNING)
    #print '\t', name, '- All %s scores obtained = ' % METRIC_FOR_TUNING, scores
    print '\t\t Avg %s: %0.3f (+/- %0.3f)' % (METRIC_FOR_TUNING, scores.mean(), scores.std() * 2)
    print '\t\t Tuned score obtained was = %0.3f' % tunedScore
    avgScores.append((name, tunedClassifier, scores.mean(), scores.std()))

#sets clf to be the best performing classifier until now
bestClassifierName = ''
bestMetric = 0.0
bestStddev = 1.0
for name, classifier, metric, stddev in avgScores:
    if metric > bestMetric:
        bestMetric = metric
        bestClassifierName = name
        bestStddev = stddev
        clf = classifier
    elif metric == bestMetric and stddev < bestStddev:
        bestMetric = metric
        bestClassifierName = name
        bestStddev = stddev
        clf = classifier

print '\t----------------------------------------------------------------------'
print '\tBest classifier was', bestClassifierName, 'with an avg %s of ' % METRIC_FOR_TUNING, bestMetric
print '\t----------------------------------------------------------------------'

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)