import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json

seed = 13
np.random.seed(seed)

# load data
df = pd.read_csv("iris.csv")
X = df.values[:, :4].astype(float)
Y = df.values[:, 4]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_one_hot = np_utils.to_categorical(Y_encoded)


def baseline_model():
    model = Sequential()
    model.add(Dense(7, input_dim=4, activation="tanh"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=1, verbose=1)


# evaluate
kfold: KFold = KFold(n_splits=10, shuffle=True, random_state=seed)
result = cross_val_score(estimator, X, Y_one_hot, cv=kfold)
print("means %.2f, std %.2f" % (result.mean(), result.std()))


#save model
estimator.fit(X, Y_one_hot)
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

estimator.model.save_weights("model.h5")
print("saved model to disk")

# load model and use it for prediction
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")

predicted = loaded_model.predict(X)
print("predicted probability:" + str(predicted))

predicted_label = loaded_model.predict_proba(X)
print("predicted label:" + str(predicted_label))















