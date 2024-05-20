import pandas as pd

data =pd.read_csv('BankNote_Authentication.csv')

y = data['class']
X = data.drop(['class'],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=42)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

print("score {}".format(score))

#create pickle file using serialization
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()
