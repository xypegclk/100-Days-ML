import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit([1, 2, 2, 6])
#print(le.classes_)
# print(le.transform([1, 1, 2, 6]))
# print(le.inverse_transform([0,0,1,2]))


le.fit(["paris", "paris", "tokyo", "amsterdam"])
# print(le.transform(["tokyo", "tokyo", "paris"]))
# print(le.inverse_transform([2,2,1]))


country = ['Taiwan', 'Australia', 'Ireland', 'Australia', 'Ireland', 'Taiwan']
age = [25, 30, 45, 35, 22, 36]
salary = [20000, 32000, 59000, 60000, 43000, 52000]
dic = {'Country': country, 'Age': age, 'Salary': salary}
data = pd.DataFrame(dic)

data['Country'] = le.fit_transform(data['Country'])