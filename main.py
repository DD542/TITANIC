import pandas as pd

#nous sert a charge directement nos donne de mon ficher data
train =pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#maintenant j exploire ses donne
print("===DIMENSION===")
print("train",train)
print("test",test)

print("\n===PREMIERE LIGNE===")
print(train.head())

print("\n ===INFO GENERALES===")
print(train.info())

print("\n ===VALEUR MANQUANTES TRAIN===")
print(train.isnull().sum())

print("\n=== VALEURS MANQUANTES TEST ===")
print(test.isnull().sum())

print("\n ===STATISTIQUE===")
print(train.describe())

#supprime la colonne cabin cas bcp de valeur manquante
train =train.drop(columns=['Cabin'])
test=test.drop(columns=['Cabin'])

print("la colonne Cabin supprime")
print("colonnes restantes:",train.columns.tolist())
print("colonnes restantes:",test.columns.tolist())

#je remplace maintenant les valeurs manquantes

age_median =train["Age"].median()
fare_median =train["Fare"].median()
embarked_mode=train["Embarked"].mode()[0]

#commencons par ange et je le base sur le principe de la mediane pour remplace les zone libre par la mediane
train["Age"]=train["Age"].fillna(age_median)
test["Age"]=test["Age"].fillna(age_median)

#par la suite je passe a Embarked--> mode
train["Embarked"]=train["Embarked"].fillna(embarked_mode)

#fare j utilise la mediane pour complete
test["Fare"]=test["Fare"].fillna(fare_median)

#verification
print("#\n =====VALEURS MANQUANTES APRES LE NETOYYAGE====")
print("===TRAIN===")
print(train.isnull().sum())
print("\n===TEST===")
print(test.isnull().sum())

#encodage sex par deux valeur normal binaire
train["Sex"]=train["Sex"].map({"male":1,"female":0})
test["Sex"]=test["Sex"].map({"male":1,"female":0})

#one-short enconding rpour embarker
train = pd.get_dummies(train, columns=['Embarked'], dtype=int)
test = pd.get_dummies(test, columns=['Embarked'], dtype=int)

#verification
print("\n===COLONNES APRES ENCODAGE====")
print("train",train.columns.tolist())
print("test",test.columns.tolist())
print("\ntrain head:")
print(train.head())
print("\ntest head:")
print(test.head())

print("\n===VERIFIONS LA CORRELATION===")
print(train.corr(numeric_only=True)['Survived'].sort_values(ascending=False))

