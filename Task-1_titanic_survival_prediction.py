import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Constants for column names
PCLASS = "Pclass"
SEX = "Sex"
AGE = "Age"
SIBSP = "SibSp"
PARCH = "Parch"
FARE = "Fare"
EMBARKED = "Embarked"
SURVIVED = "Survived"

data = pd.read_csv("tested.csv")

def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df[AGE].fillna(df[AGE].mean(), inplace=True)
    df[EMBARKED].fillna(df[EMBARKED].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=[SEX, EMBARKED], drop_first=True)
    return df

data = preprocess_data(data)

X = data.drop(columns=[SURVIVED])
y = data[SURVIVED]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

def predict_survival():
    print("Please provide passenger information:")
    try:
        pclass = int(input(f"{PCLASS} (1, 2, 3): "))
        sex = input(f"{SEX} (male or female): ")
        age = float(input(f"{AGE}: "))
        sib_sp = int(input(f"{SIBSP}: "))
        parch = int(input(f"{PARCH}: "))
        fare = float(input(f"{FARE}: "))
        embarked = input(f"{EMBARKED} (C, Q, or S): ")

        user_input = pd.DataFrame(
            {
                PCLASS: [pclass],
                f"{SEX}_male": [1 if sex.lower() == "male" else 0],
                f"{EMBARKED}_Q": [1 if embarked.upper() == "Q" else 0],
                f"{EMBARKED}_S": [1 if embarked.upper() == "S" else 0],
                AGE: [age],
                SIBSP: [sib_sp],
                PARCH: [parch],
                FARE: [fare],
            }
        )

        user_input = user_input[X_train.columns]
        prediction = model.predict_proba(user_input)

        plt.figure(figsize=(8, 5))
        plt.bar(["Not Survived", "Survived"], prediction[0], color=["red", "green"])
        plt.xlabel("Survival Prediction")
        plt.ylabel("Prediction Probability")
        plt.title("Passenger Survival Prediction Chances")
        plt.ylim(0, 1)

        if prediction[0][0] > prediction[0][1]:
            print("The passenger did not survive.")
        else:
            print("The passenger survived.")

        plt.show()
    except ValueError:
        print("Invalid input. Please enter valid values.")

predict_survival()
