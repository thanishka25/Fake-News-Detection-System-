import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


fake = pd.read_csv("dataset/Fake.csv")

real = pd.read_csv("dataset/True.csv")


fake["label"] = 0

real["label"] = 1


data = pd.concat([fake, real])


X = data["text"]

y = data["label"]


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2
)


model = LogisticRegression()


model.fit(X_train, y_train)


prediction = model.predict(X_test)


print(
"Accuracy:",
accuracy_score(
y_test,
prediction
)
)