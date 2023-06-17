#Importar librerías
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Cargar los datos de los archivos proporcionados por Kaggle
train = r'c :\Users\elton\Downloads\Ale\6to semestre\investigacion de operaciones\Ciencias de datos en Python\train.csv'
test =  r'c :\Users\elton\Downloads\Ale\6to semestre\investigacion de operaciones\Ciencias de datos en Python\test.csv'

train_file = 'train.csv'
test_file = 'test.csv'
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

#Verificar los datos
print("\nDatos de entrenamiento:\n ")
print(df_train.head())
print(df_train.info())
print(df_train.isnull().sum())

print("\nDatos de prueba:\n")
print(df_test.head())
print(df_test.info())
print(df_test.isnull().sum())

#Preprocesamiento de datos
df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train['Sex'] = df_train['Sex'].map({'male': 1, 'female': 0})
df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})
df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_train['Age'] = pd.cut(df_train['Age'], bins=[0, 8, 15, 18, 25, 40, 60, 100], labels=['1', '2', '3', '4', '5', '6', '7'])
df_test['Age'] = pd.cut(df_test['Age'], bins=[0, 8, 15, 18, 25, 40, 60, 100], labels=['1', '2', '3', '4', '5', '6', '7'])
df_train.dropna(inplace=True)

#Entrenamiento y Prueba de datos
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crear Modelos de aprendizaje automático
#Regresión Logística
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("\nPrecisión Regresión Logística:\n"), 
print (log_reg.score(X_test, y_test))

#SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("\nPrecisión SVM:\n")
print (svm.score(X_test, y_test))

#K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\nPrecisión KNN:\n")
print(knn.score(X_test, y_test))

#Predicción con modelos
test_ids = df_test['PassengerId']
test_data = df_test.drop('PassengerId', axis=1)
test_data.fillna(test_data.mean(), inplace=True)

log_reg_pred = log_reg.predict(test_data)
svm_pred = svm.predict(test_data)
knn_pred = knn.predict(test_data)

log_reg_output = pd.DataFrame({'PassengerId': test_ids, 'Survived': log_reg_pred})
svm_output = pd.DataFrame({'PassengerId': test_ids, 'Survived': svm_pred})
knn_output = pd.DataFrame({'PassengerId': test_ids, 'Survived': knn_pred})

print("\n\nPredicción Regresión Logística:")
print(log_reg_output.head())
print("\n\nPredicción SVM:")
print(svm_output.head())
print("\n\nPredicción KNN:")
print(knn_output.head())