import numpy as np
import pandas as pd
import statsmodels.api as sm
from time import time

TRAINPATH = "C:\Python27\Kaggle\\titanic\\train.csv"
TESTPATH = "C:\Python27\Kaggle\\titanic\\test.csv"

def random_forest(train_df,test_df):
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators = 100,min_samples_split = 40)
	features_train = train_df[0::,1::]
	labels_train = train_df[0::,0]
	features_test = test_df[0::,1::]
	labels_test = test_df[0::,0]

	#fit model
	t0 = time()
	clf.fit(features_train,labels_train)
	print "training time RF:", round(time()-t0,3), "s"

	#predict new data
	t0 = time()
	pred = clf.predict(features_test)
	print "prediction time RF:", round(time()-t0,3), "s"

	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(labels_test, pred)
	return accuracy

def SVMAccuracy(train_df,test_df):
	from sklearn.svm import SVC
	clf = SVC(C=10000.,kernel="rbf",gamma = 3.0)
	features_train = train_df[0::,1::]
	labels_train = train_df[0::,0]
	features_test = test_df[0::,1::]
	labels_test = test_df[0::,0]

	t0 = time()
	clf.fit(features_train,labels_train)
	print "training time SVM:", round(time()-t0,3), "s"

	t0 = time()
	pred = clf.predict(features_test)
	print "prediction time SVM:", round(time()-t0,3), "s"

	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(labels_test, pred)
	return accuracy

def data_clean(file_path):
	df = pd.read_csv(file_path,header =0)
	df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	#if len(df.Embarked[df.Embarked.isnull()]) > 0:
	#	df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values

	#Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
	#Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
	#df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

	# All the ages with no data -> make the median of all Ages
	median_age = df['Age'].dropna().median()
	if len(df.Age[df.Age.isnull()]) > 0:
		df.loc[(df.Age.isnull()), 'Age'] = median_age

	passengerIds = df['PassengerId']
	# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
	df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Fare','Embarked', 'PassengerId'], axis=1) 

	return df, passengerIds

def main():
	train_df, passengerIds = data_clean(TRAINPATH)
	training_df, test_df = train_df[:700], train_df[700:]
	train_data = training_df.values
	test_data = test_df.values
	print random_forest(train_data,test_data)
	
	
if __name__ == '__main__':
	main()
