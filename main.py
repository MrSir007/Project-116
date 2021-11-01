import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as acs

getData = pd.read_csv("Admission_Predict.csv")
TOEFLPop = getData["TOEFL Score"].tolist()
GREPop = getData["GRE Score"].tolist()
chancePop = getData["Chance of admit"].tolist()

colorlist = []
for i in chancePop :
  if i == 1 :
    colorlist.append("green")
  else :
    colorlist.append("red")
scatter = pgo.Figure(data=pgo.Scatter(x=TOEFLPop,y=GREPop,mode="markers",marker=dict(color=colorlist)))
'''scatter.show()'''

scoreData = getData[["TOEFL Score","GRE Score"]]
resultData = getData["Chance of admit"]

scoreTrain, scoreTest, resultTrain, resultTest = tts(scoreData,resultData,test_size=0.25,random_state=0)

scx = ss()
scoreTrain = scx.fit_transform(scoreTrain)
scoreTest = scx.fit_transform(scoreTest)

classifier = lr(random_state=0)
classifier.fit(scoreTrain,resultTrain)
prediction = classifier.predict(scoreTest)
accuracy = acs(resultTest,prediction)
'''print(purchaseAccuracy)'''

TOEFLInput = int(input("TOEFL Score "))
GREInput = int(input("GRE Score "))
userTest = scx.transform([[TOEFLInput,GREInput]])
userPurchasePrediction = classifier.predict(userTest)
if userPurchasePrediction [0]==1 :
  print("The person may be admitted")
else :
  print("The person may not be admitted")