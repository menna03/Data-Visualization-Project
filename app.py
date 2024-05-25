from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import pandas as pd
import pickle
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', message=None)

@app.route('/train', methods=['POST'])
def train_and_save_model():
    # Create a DataFrame from the data
    csv_file = request.files['file']

    # Read the CSV file into a DataFrame
    df = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include='object').columns
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Selecting features and target variable
    # Selecting features and target variable
    X = df[
        ['Month',
         'Age',
         'Occupation',
         'Annual_Income',
         'Monthly_Inhand_Salary',
         'Num_Bank_Accounts',
         'Num_Credit_Card',
         'Interest_Rate',
         'Type_of_Loan',
         'Delay_from_due_date',
         'Num_of_Delayed_Payment',
         'Changed_Credit_Limit',
         'Num_Credit_Inquiries',
         'Credit_Mix',
         'Outstanding_Debt',
         'Total_EMI_per_month',
         'Total_Months']
    ]  # Features
    y = df["Credit_Score"]  # Target variable

    # Initialize CatBoostClassifier
    catboost_clf = CatBoostClassifier()

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit CatBoostClassifier on training data
    catboost_clf.fit(X_train, y_train)

    # Save the trained model as a pickle file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(catboost_clf, f)

    message = 'Model trained successfully and saved as trained_model.pkl file'
    return jsonify({'message': message})

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Read the data from the form
    data = {}
    for key, value in request.form.items():
        data[key] = [value]

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Encode categorical features
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    # Predict
    prediction = model.predict(df)

    # Convert prediction to a native Python data type
    prediction = int(prediction)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
