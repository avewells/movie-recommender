from pyspark import SparkContext
import sys
import math
import numpy as np
from scipy.stats import pearsonr
import timeit
from pyspark.mllib.recommendation import ALS
import warnings
warnings.filterwarnings("ignore")


#set up spark context and input file and output files
sc = SparkContext("local", "Movie Ratings")
trainingFile = 'HW5train.txt' 
validationFile = 'HW5valid.txt'
testingFile = 'HW5test.txt' 
#trainingFile = 'training.dat'
#validationFile = 'validation.dat'
#testingFile = 'testing.dat'

outFile = open('output.txt', 'w')

#set num of users and movies
numUsers = 6040
numMovies = 3952
#numUsers = 12
#numMovies = 6


#used to format each line into specific format during map
def get_ratings_tuple(entry):
	items = entry.split('::') #!!!!!! this needs to be changed for each data set
	return int(items[0]), int(items[1]), float(items[2])#, int(items[3])


#used to format each line into specific format during map	
def get_movie_tuple(entry):
	items = entry.split('::') #!!!!!! this needs to be changed for each data set
	return int(items[0]), items[1]


#sort function from the reference	
def sortFunction(tuple):
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

    
#helper function that calcs the avg rating and number of ratings for a tuple
#goes from: MovieID, (Rating1, Rating2, Rating3, ...)) to (MovieID, (number of ratings, averageRating))
def getCountsAndAverages(IDandRatingsTuple):
    numRatings = len(IDandRatingsTuple[1])
    sumRatings = sum(IDandRatingsTuple[1])
    return IDandRatingsTuple[0], (int(numRatings), float(sumRatings)/float(numRatings))

    
#helper function to compute RMSE
def computeError(predictedRDD, actualRDD):
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda entry: ((entry[0], entry[1]), entry[2]))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda entry: ((entry[0], entry[1]), entry[2]))

    # Compute the squared error for each matching entry 
    squaredErrorsRDD = predictedReformattedRDD.join(actualReformattedRDD).map(lambda entry: (entry[1][0] - entry[1][1])**2)

    # Compute the total squared error
    totalError = squaredErrorsRDD.reduce(lambda x,y: x + y)

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(float(totalError) / float(numRatings))
    
    
def cf_model():
	#set up the training, validation, and testing RDDs
	trainingRDD = sc.textFile(trainingFile).map(get_ratings_tuple).cache()
	validationRDD = sc.textFile(validationFile).map(get_ratings_tuple).cache()
	testingRDD = sc.textFile(testingFile).map(get_ratings_tuple).cache()

	#extracted pairs
	validationForPredictRDD = validationRDD.map(lambda entry: (entry[0], entry[1]))
	trainingForPredictRDD = trainingRDD.map(lambda entry: (entry[0], entry[1]))

	#training the model and finding the best rank from validation set (from reference)
	seed = 5L
	iterations = 5
	regularizationParameter = 0.1
	ranks = [4, 8, 12]
	errors = [0, 0, 0]
	err = 0
	tolerance = 0.03

	minError = float('inf')
	bestRank = -1
	bestIteration = -1
	for rank in ranks:
		#train the model and time it
		startTime = timeit.default_timer()
		model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations, lambda_=regularizationParameter)
		elapsedTime = timeit.default_timer() - startTime
		outFile.write('Training Time: %f\n' % elapsedTime)
	
		#compute training error
		predictedTrainingRDD = model.predictAll(trainingForPredictRDD)
		trainError = computeError(predictedTrainingRDD, trainingRDD)
		outFile.write('Training RMSE: %s\n' % trainError)
	
		#test with the validation set and time it
		startTime = timeit.default_timer()
		predictedRatingsRDD = model.predictAll(validationForPredictRDD)
		elapsedTime = timeit.default_timer() - startTime
		outFile.write('Validation Time: %f\n' % elapsedTime)
	
		#compute error on validation set and record best
		error = computeError(predictedRatingsRDD, validationRDD)
		errors[err] = error
		err += 1
		outFile.write('For rank %s the RMSE is %s\n' % (rank, error))
		if error < minError:
			minError = error
			bestRank = rank

	outFile.write('The best model was trained with rank %s\n' % bestRank)

	#testing best rank on test set (based on reference)
	myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations, lambda_=regularizationParameter)
	testForPredictingRDD = testingRDD.map(lambda entry: (entry[0], entry[1]))

	#time the test
	startTime = timeit.default_timer()
	predictedTestRDD = myModel.predictAll(testForPredictingRDD)
	elapsedTime = timeit.default_timer() - startTime
	outFile.write('Testing Time: %f\n' % elapsedTime)

	testRMSE = computeError(testingRDD, predictedTestRDD)

	outFile.write('The model had a RMSE on the test set of %s\n' % testRMSE)


#reads in values from training file and initialize matrix
def readValues(trainingFile):
	ratingsMatrix = np.zeros(shape=(numMovies,numUsers))
	with open(trainingFile) as train:
		for line in train:
			items = line.split('::')
			user = int(items[0])
			movie = int(items[1])
			rating = float(items[2])
			ratingsMatrix[movie-1, user-1] = rating
			
	return ratingsMatrix

	
#calculates rmse between estimated matrix and true values
def calc_error(est, true):
	return np.sqrt(((est[np.nonzero(true)] - true[np.nonzero(true)]) ** 2).mean())

	
#baseline estimate
def cf_baseline():
	#get training and test matrices
	trainingMatrix = readValues(trainingFile)
	testMatrix = readValues(testingFile)

	#get overall avg
	avgRating = np.mean(trainingMatrix[np.nonzero(trainingMatrix)])
	
	#get avg for each movie
	movAvg = np.apply_along_axis(lambda x: np.mean(x[np.nonzero(x)]), 1, trainingMatrix)
	movAvg[np.isnan(movAvg)] = avgRating
	
	#get avg for each user
	usrAvg = np.apply_along_axis(lambda x: np.mean(x[np.nonzero(x)]), 0, trainingMatrix)
	usrAvg[np.isnan(usrAvg)] = avgRating
	
	#predict for all missing ratings
	for i, v in np.ndenumerate(trainingMatrix):
		if v == 0.0:
			trainingMatrix[i] = avgRating + (movAvg[i[0]] - avgRating) + (usrAvg[i[1]] - avgRating)
	
	#calculate test error
	testError = calc_error(trainingMatrix, testMatrix)
	print "Baseline Estimate: " + str(testError)
	
	#save matrix for later use
	np.save('baseline', trainingMatrix)
	

#user-user filtering
def cf_userUser():
	#get training, validation, and test matrices
	trainingMatrix = readValues(trainingFile)
	validMatrix = readValues(validationFile)
	testMatrix = readValues(testingFile)
	
	#get previously saved matrices
	#usrSim = np.load('user_user_full.npy')
	baseline = np.load('baseline_full.npy')
	
	#calculate similarity matrix for all users
	t_usrSim = np.zeros((numUsers, numUsers))
	for id1 in xrange(numUsers):
		for id2 in xrange(id1, numUsers):
			t_usrSim[id1, id2] = pearsonr(trainingMatrix[:,id1], trainingMatrix[:,id2])[0]
	t_usrSim[np.isnan(t_usrSim)] = 0.0
	usrSim = np.triu(t_usrSim) + np.triu(t_usrSim, -1).T
	
	#save similarity matrix so other parameters can be quickly tested without recalculating each time
	#np.save('user_user', usrSim)
	
	#predict for all missing ratings using N most similar users
	N = 10 #change this to get different models
	row, col = np.nonzero(testMatrix) #change for validation vs. test
	for i in xrange(len(row)):
		#get N most similar
		mostSim = []
		simInd = [j[0] for j in sorted(enumerate(usrSim[:,col[i]]), key=lambda x:-x[1])]
		for ind in simInd:
			if trainingMatrix[row[i], ind] != 0 and ind != col[i]:
				mostSim.append(ind)
			if len(mostSim) == N:
				break
		sum1 = 0.0
		sum2 = 0.0
		
		for ind in mostSim:
			if ind != col[i]:
				sum1 += usrSim[col[i], ind] * trainingMatrix[row[i], ind]
				sum2 += usrSim[col[i], ind]
		if sum2 == 0.0:
			trainingMatrix[row[i], col[i]] = baseline[row[i], col[i]]
		else:
			trainingMatrix[row[i], col[i]] = sum1 / sum2
	
	#calculate validation error
	validError = calc_error(trainingMatrix, validMatrix)
	print "User-User Validation Error: " + str(validError)
	
	#calculate test error
	testError = calc_error(trainingMatrix, testMatrix)
	print "User-User Test Error: " + str(testError)
	
	
#item-item filtering
def cf_itemItem():
	#get training, validation, and test matrices
	trainingMatrix = readValues(trainingFile)
	validMatrix = readValues(validationFile)
	testMatrix = readValues(testingFile)
	
	#get previously saved matrices
	#movSim = np.load('item_item_full.npy')
	baseline = np.load('baseline_full.npy')
	
	#calculate similarity matrix for all items
	t_movSim = np.zeros((numMovies, numMovies))
	
	for id1 in xrange(numMovies):
		for id2 in xrange(id1, numMovies):
			t_movSim[id1, id2] = pearsonr(trainingMatrix[id1,:], trainingMatrix[id2,:])[0]
	t_movSim[np.isnan(t_movSim)] = 0.0
	movSim = np.triu(t_movSim) + np.triu(t_movSim, -1).T
	
	#save similarity matrix so other parameters can be quickly tested without recalculating each time
	#np.save('item_item', movSim)
	
	#predict for all missing ratings using N most similar movies
	N = 10 #change this to get different models
	row, col = np.nonzero(testMatrix) #change this for validation vs. test
	for i in xrange(len(row)):
		#get N most similar
		mostSim = []
		simInd = [j[0] for j in sorted(enumerate(movSim[:,row[i]]), key=lambda x:-x[1])]
		for ind in simInd:
			if trainingMatrix[ind, col[i]] != 0 and ind != row[i]:
				mostSim.append(ind)
			if len(mostSim) == N:
				break
		sum1 = 0.0
		sum2 = 0.0
		
		for ind in mostSim:
			if ind != row[i]:# and movSim[ind, row[i]] > 0:
				sum1 += movSim[ind, row[i]] * trainingMatrix[ind, col[i]]
				sum2 += movSim[ind, row[i]]
		if sum2 == 0.0:
			trainingMatrix[row[i], col[i]] = baseline[row[i], col[i]]
		else:
			trainingMatrix[row[i], col[i]] = sum1 / sum2
	
	#calculate validation error
	validError = calc_error(trainingMatrix, validMatrix)
	print "Item-Item Validation Error: " + str(validError)
	
	#calculate test error
	testError = calc_error(trainingMatrix, testMatrix)
	print "Item-Item Test Error: " + str(testError)
	

#do baseline estimate
cf_baseline()

#do user-user filtering
cf_userUser()

do item-item filtering
cf_itemItem()

#model based CF
cf_model()

