import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def preprocess_data(df):
    label_encoder = LabelEncoder()
    df["species"] = label_encoder.fit_transform(df["species"])
    return df

def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

def get_user_input():
    print("Please provide iris measurements:")
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))
    return pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

def main():
    data = pd.read_csv("IRIS.csv")
    data = preprocess_data(data)
    X = data.drop(columns=["species"])
    y = data["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    user_input = get_user_input()
    prediction = model.predict(user_input)
    probabilities = model.predict_proba(user_input)[0]
    
    species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predicted_species = species_names[prediction[0]]
    print(f"Predicted Iris Species: {predicted_species}")
    
    plt.bar(species_names, probabilities, color=['blue', 'violet', 'yellow'])
    plt.xlabel('Iris Species')
    plt.ylabel('Probability')
    plt.title('Probability of Each Iris Species')
    plt.show()

if __name__ == "__main__":
    main()
