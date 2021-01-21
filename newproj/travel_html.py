# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:17:36 2019

@author: sheki
"""
import enum
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os, pickle



df = pd.read_csv("website_db.csv", encoding="ISO-8859-1")
def fileter(state, ntrav, fav_cusine, age, budget):
    themes = [[
"Pilgrimage",
"Wildlife", 
"Historic",
"Hill Station",
"Wildlife",
"Scenic",
"Adventurous",
"Harbour",
"Water Sports"
]]
    df = pd.read_csv("website_db.csv", encoding="ISO-8859-1")
    budget = int(budget)
    df1 = df[((df["Age"] < (age+5)) | (df["Age"] > (age-3)))]
    if df1.shape[0] == 0:
        return df
    else:
        df2 = df1[((df1["p_ub"] >= (budget+10000)) | ((df1["p_ub"] <= (budget-10000))))]
        if df2.shape[0] == 0:
            return df1
        else:
            df3 = df2[df2["Preferred_cuisines"] == fav_cusine]
            if df3.shape[0] == 0:
                return df2
            else:
                return df3

def get_index_from_id(train, theme):
    return train[train["User_Id"] == theme].index.tolist()[0]


def print_similar_places(train, indices, query=None):
    found_id = get_index_from_id(train, query)
    l = []
    for id in indices[found_id][1:]:
        q = []
        q.append(train.iloc[id]["Age"])
        q.append(train.iloc[id]["Place"])
        q.append(train.iloc[id]["State"])
        q.append(train.iloc[id]["Preferred_cuisines"])
        q.append(train.iloc[id]["Images"])
        q.append(train.iloc[id]["Description"])
        l.append(q)
        l.sort(key=lambda x: x[2], reverse=True)
    return l


def allinall():
    df = pd.read_csv("website_db.csv", encoding="ISO-8859-1")
    df["Images"] = df["Place"] + ".jpg"

    train = df[
        [
            "User_Id",
            "Age",
            "Gender_code",
            "User_Locale",
            "Preferred_cuisines",
            "Place",
            "State",
            "Theme",
            "Number_of_travellers",
            "Images",
            "Description",
        ]
    ]
    features = pd.concat(
        [
            pd.get_dummies(train[["Theme"]]),
            pd.get_dummies(train[["Gender_code"]]),
            pd.get_dummies(train[["User_Locale"]]),
            pd.get_dummies(train[["Preferred_cuisines"]]),
            pd.get_dummies(train[["Place"]]),
            pd.get_dummies(train[["State"]]),
        ],
        axis=1,
    )
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)

    model = NearestNeighbors(n_neighbors=20, metric="cosine")
    with open("model_pickle", "wb") as f:
        pickle.dump(model, f)
    nbrs = model.fit(features)
    distances, indices = nbrs.kneighbors(features)

    all_state_names = list(train.State.values)

    rec = print_similar_places(train=train, indices=indices, query=1002)
    rec = pd.DataFrame(rec).drop_duplicates(subset=[1, 2])
    rec = rec.drop(0)
    
    output = {
    "df": df,
    "train": train,
    "rec" : rec,
    "distances": distances,
    "indices": indices
    }

    return output

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///robin.db"

db=SQLAlchemy(app)

class Gender(enum.Enum):
    male=0
    female=1

class robin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(200), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    phoneNo = db.Column(db.String(10), nullable=False)
    locale = db.Column(db.String(200), nullable=False)
    Dob = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.Enum(Gender), nullable=False)
    Age = db.Column(db.Integer)
    reg_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route("/", methods = ["GET", "POST"])
def login():
	if request.method == "POST":
		username = request.form["username"]
		password = request.form["password"]
		print(username)
		print(password)
		user_check = robin.query.filter_by(user_name = username).all()
		user_check = robin.query.filter_by(password = password).all()
		print(user_check)
		print("above line is user check")
		if len(user_check) == 0:
			return redirect("/")
		else:
			session["username"] = username
			return redirect(url_for('home'))
	else:
		return render_template('login.html')

@app.route("/registration", methods = ["GET", "POST"])
def registration():
	if request.method == "POST":
		user_name = request.form["username"]
		password = request.form["password1"]
		confirm_password = request.form["confirm-password1"]
		email = request.form["email"]
		phoneNo = request.form["phno"]
		locale = request.form["locale"]
		Dob = request.form["date1"]
		gender = request.form["gender"]
		Age = request.form["age"]
		if password != confirm_password:
			return "password do not match"
		print("{}, {}, {}, {}, {}, {}, {}, {}".format(
			user_name,password,email,phoneNo,locale,Dob,gender,Age
			))
		new_user = robin(
			user_name = user_name,
			password = password,
			email = email,
			phoneNo = phoneNo,
			locale = locale,
			Dob = Dob,
			gender = gender,
			Age = Age
			)
		db.session.add(new_user)
		db.session.commit()
		try:
			db.session.commit()
			return redirect("/")
		except:
			return "Error"
	return render_template('registration.html')
    
@app.route("/home", methods = ["GET", "POST"])
def home():
        rec = allinall()
        print(rec)
        rec = rec["rec"]
        rec = rec[[1, 2, 4, 5]]
        display_dict = rec.to_dict("records")
        return render_template("home.html", display_dict=display_dict)

@app.route("/search", methods = ["GET", "POST"])
def search():
    fav_cusine = request.form["fav-cusine"]
    num_travellers = request.form["ntrav"]
    state = request.form["state"]
    budget = request.form["budget_range"]
    output = allinall()
    df = output["train"]
    username = session.get("username")
    user_info = robin.query.filter_by(user_name = username).all()
    user_info = user_info[0]
    age = user_info.Age
    df = fileter(state=state, ntrav=num_travellers, fav_cusine=fav_cusine, age=age, budget=budget)
    if df.shape[0] > 0:
        states = list(df["State"].unique())
        df = df.sort_values(by = "p_ub")
        df["Image"] = df["Place"] + ".jpg"
        df = df[["Place", "Image","Description"]]
    seed_val= states.index(state)
    np.random.seed(seed_val)
    rows = np.random.randint(0, df.shape[0], size = 5)
    print(rows)
    output_df = pd.DataFrame()
    for i in rows:
        output_df = output_df.append(df.iloc[i])
        
    display_dict = output_df.to_dict("records")
    return render_template("recommendation.html", display_dict=display_dict)




    
@app.route("/age")
def age():
    age_freq = (
        df.drop_duplicates(subset="User_Id")
        .groupby("Age")["Age"]
        .count()
        .reset_index(name="freq")
    )
    age = list(age_freq["Age"])
    freq = list(age_freq["freq"])
    return render_template("age.html", age=age, freq=freq)


@app.route("/gender")
def gender():
    gender_freq = (
        df.drop_duplicates(subset="User_Id")
        .groupby("Gender_code")["Gender_code"]
        .count()
        .reset_index(name="freq")
    )
    gender = list(gender_freq["Gender_code"])
    freq = list(gender_freq["freq"])
    return render_template("gender.html", gender=gender, freq=freq)


@app.route("/month")
def month():
    df["Year"] = pd.DatetimeIndex(df["Date"]).year
    df["Month"] = pd.DatetimeIndex(df["Date"]).month
    month_freq = df.groupby(["Month"])["Month"].count().reset_index(name="freq")
    month = list(month_freq["Month"])
    freq = list(month_freq["freq"])
    return render_template("month.html", month=month, freq=freq)


@app.route("/ad")
def ad():
    ad_freq = (
        df.drop_duplicates(subset="User_Id")
        .groupby("Ad_source")["Ad_source"]
        .count()
        .reset_index(name="freq")
    )
    ad = list(ad_freq["Ad_source"])
    freq = list(ad_freq["freq"])
    return render_template("ad.html", ad=ad, freq=freq)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")




if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)
