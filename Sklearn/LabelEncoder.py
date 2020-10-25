from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit([1, 2, 2, 6])
#print(le.classes_)
print(le.transform([1, 1, 2, 6]))
