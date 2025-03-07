import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

search_params = False
test_model = False

file_path = 'TrainOnMe.csv'
data = pd.read_csv(file_path)

y_column = data.columns[1]
x_columns = data.columns[2:]

x_train = data[x_columns]
y_train = data[y_column]

categorical_columns = ["x7", "x12"]

preprosessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
], remainder = 'passthrough')

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

if test_model:
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    if search_params:
        model = Pipeline([
            ('preprosessor', preprosessor),
            ('classifier', XGBClassifier(random_state=42))
        ])

        param_grid = {
            'classifier__n_estimators': [400, 500, 600, 700],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [2, 3, 4] 
        }

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        if hasattr(grid_search, "best_estimator_"):
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Best model accuracy: {accuracy:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            print("Grid search did not complete successfully")
    else:
        model = Pipeline([
            ('preprosessor', preprosessor),
            ('classifier', XGBClassifier(learning_rate=0.05, max_depth=2, n_estimators=600, random_state=42))
        ])

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
else:
    model = Pipeline([
        ('preprosessor', preprosessor),
        ('classifier', XGBClassifier(learning_rate=0.05, max_depth=2, n_estimators=600, random_state=42))
    ])

    model.fit(x_train, y_train)

    eval_file_path = "EvaluateOnMe.csv"
    eval_data = pd.read_csv(eval_file_path)
    x_eval = eval_data[x_columns]

    y_eval_pred = model.predict(x_eval)
    y_eval_pred_labels = label_encoder.inverse_transform(y_eval_pred)

    output_file = "predictions.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(y_eval_pred_labels))
    print(f"Predictions saved to {output_file}")