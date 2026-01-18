import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


# Load Dataset 
df = pd.read_csv("water_potability.csv")
df.head()


X = df.drop('Potability',axis=1)
y = df['Potability']



numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', RobustScaler())
])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)


svc_pipe = Pipeline(
    [
        ('numercal_pipe',num_pipeline),
        ('model',SVC(class_weight='balanced',probability=True))
    ]
)


param_grid = {
              'model__C': [0.01,0.1,1,10,100],
              'model__max_iter': [1000,5000]
              }



grid_search = GridSearchCV(
    estimator=svc_pipe,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train,y_train)


print("Best Parameters Found:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


best_lr_model = grid_search.best_estimator_

y_pred = best_lr_model.predict(X_test)
print("Final Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)



# Save the trained model as a pickle file
model_filename = "svc_water_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_lr_model, file)

print(f"Trained model saved as {model_filename}")
