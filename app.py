from flask import Flask,render_template,request
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
df = pd.read_csv("E://Datasets//Titanic-Dataset.csv")
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df2 = df.drop(columns=['PassengerId'],axis=1)
df2 = df2.drop(columns = ['Name',"Cabin","Fare","Ticket"],axis=1)
le = LabelEncoder()
cols = ['Sex','Embarked']
for c in cols:
    df2[c] = le.fit_transform(df2[c])
train_df = df2.sample(frac=0.7, random_state=25)
test_df = df2.drop(train_df.index)
X = df2.drop(columns=['Survived'],axis=1)
Y = df2['Survived']
def make_classify(model,arr):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=25)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred = model.predict(x_train)
    print(y_pred)
    return model.predict([arr])

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
@app.route("/ab",methods=["POST"])
def ab():
    try:
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = int(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        emb = request.form['emb']
        dic = {'S':2,'C':0,'Q':1}
        emb = dic[emb]
        dic2 = {'Male':0,'Female':1}
        sex = dic2[sex]
        arr=[pclass,sex,age,sibsp,parch,emb]
        
        print(arr)
        P = make_classify(model2,arr)[0]
        msg = {1:'SURVIVED',0:'NOT SURVIVED'}
        mp = {1:'/static/tick.jpg',0:'/static/wrong.jpg'}
        return render_template("message.html",val=mp[P],message=msg[P])
    except:
        pass
    #'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'
    msg = {1:'SURVIVED',0:'NOT SURVIVED'}
    return render_template("message.html",val='/static/tick.jpg',message="SURVIVED")
@app.route("/go_to_main",methods=["GET"])
def go_to_main():
    return render_template("index.html")
if(__name__ == '__main__'):
    app.run(debug=True)