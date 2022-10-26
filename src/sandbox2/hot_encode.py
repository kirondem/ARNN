from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(y.reshape(-1, 1))
