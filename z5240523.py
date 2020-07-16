import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso


# process cast column
def cast_handler(data):
    for index, actors in zip(data.index, data["cast"]):
        actor_list = []
        for dic in actors:
            if dic["order"] < 1:
                actor_list.append(dic["name"])
        # if len(actor_list) != 1:
        #     # print(index, actor_list)
        data.loc[index, "cast"] = str(actor_list)


def crew_handler(data):
    # process crew column
    for index, workers in zip(data.index, data["crew"]):
        worker_list = []
        for dic in workers:
            if dic["job"] == "Director":
                worker_list.append(dic["name"])
        # if len(worker_list) != 1:
        #     # print(index, worker_list)
        data.loc[index, "crew"] = str(worker_list)


def counter(value_list, df, column_name):
    value_dic = {}
    for value in value_list:
        for v in value.split(", "):
            if v in value_dic.keys():
                value_dic[v] += 1
            else:
                value_dic[v] = 1
    # # print(value_dic)
    if len(value_dic) > 30:
        value_list = sorted(value_dic.items(), key=lambda x: x[1], reverse=True)[:30]
    else:
        value_list = sorted(value_dic.items(), key=lambda x: x[1], reverse=True)
    value_list = [k[0] for k in value_list]
    for value in value_list:
        df[value] = df[column_name].str.contains(value).apply(lambda x: 1 if x else 0)


# this is a method deal with the dic in json column, return the values whose key are name
def parse_part_json_column(dic_of_list):
    name_list = []
    for dic in dic_of_list:
        name_list.append(dic["name"])
    return name_list


# remove "[]"'" in the string column
def parse_string(df, c):
    df[c] = df[c].str.strip("[]").str.replace("'", "").str.replace('"', "").str.replace(" ", "")


train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("validation.csv")

# back up the training data
data0 = train_data.copy()
data1 = test_data.copy()

# drop the missing data
data0.dropna(axis=0, how="any", inplace=True)

# convert the release_date column to year and month
data0["release_date"] = pd.to_datetime(data0["release_date"])
data1["release_date"] = pd.to_datetime(data1["release_date"])
data0["year"] = data0["release_date"].dt.year
data0["month"] = data0["release_date"].dt.month
data1["year"] = data1["release_date"].dt.year
data1["month"] = data1["release_date"].dt.month

# these columns' value are all json string which have same structure, process them together
json_columns = ["genres", "keywords", "production_countries", "spoken_languages"]

for column in json_columns:
    data0[column] = data0[column].apply(json.loads, encoding="utf-8")
    data1[column] = data1[column].apply(json.loads, encoding="utf-8")

# these columns' value are all json string and their structure is complex (nested data)
data0["crew"] = data0["crew"].apply(json.loads, encoding="utf-8")
data1["crew"] = data1["crew"].apply(json.loads, encoding="utf-8")
data0["cast"] = data0["cast"].apply(json.loads, encoding="utf-8")
data1["cast"] = data1["cast"].apply(json.loads, encoding="utf-8")
data0["production_companies"] = data0["production_companies"].apply(json.loads, encoding="utf-8")
data1["production_companies"] = data1["production_companies"].apply(json.loads, encoding="utf-8")

# covert the json data to name value
for name in json_columns:
    data0[name] = data0[name].apply(parse_part_json_column)
    data1[name] = data1[name].apply(parse_part_json_column)

data0["production_companies"] = data0["production_companies"].apply(parse_part_json_column)
data1["production_companies"] = data1["production_companies"].apply(parse_part_json_column)

# deal with the cast column
cast_handler(data0)
cast_handler(data1)

# deal with the crew column
crew_handler(data0)
crew_handler(data1)

# format the cast and crew column
parse_string(data0, "cast")
parse_string(data1, "cast")
parse_string(data0, "crew")
parse_string(data1, "crew")

# parse list columns
list_columns = ["genres", "keywords", "production_countries", "production_companies", "spoken_languages"]
for column in list_columns:
    data0[column] = data0[column].apply(lambda x: ", ".join(map(str, x)))
    data1[column] = data1[column].apply(lambda x: ", ".join(map(str, x)))

# get top 30 value in these columns and one hot encoding
one_zero_column = ["genres", "cast", "crew", "spoken_languages"]
for i in one_zero_column:
    counter(list(data0[i]), data0, i)
    counter(list(data1[i]), data1, i)


# split to train data x and train data y, test data x, test data y

# y_value is the revenue column
train_y = data0["revenue"]
test_y = data1["revenue"]

# drop the text columns
train_x = data0.drop(["movie_id", "cast", "crew", "genres", "homepage", "keywords", "original_language", "original_title",
                      "overview", "production_companies", "production_countries", "release_date", "spoken_languages",
                      "status", "tagline", "rating"], axis=1)

# use the correlation to choose columns
corr = train_x.corr()
test = corr["revenue"].sort_values()
result = np.where((test < -0.1) | (test > 0.1))
column_name = list(test.index)
keep_columns = [column_name[i] for i in list(result[0])]

# keep the columns which has the correlation > 0.1, <-0.1 with revenue
train_x = data0[keep_columns]

# drop the y column
train_x = train_x.drop(["revenue"], axis=1)

# for test x, find columns which both appear in train data frame and test data frame
test_x = data1.copy()
train_column0 = list(train_x)
train_column1 = list(test_x)
train_column = [i for i in train_column0 if i in train_column1]
remove0 = [i for i in train_column0 if i not in train_column]
remove1 = [i for i in train_column1 if i not in train_column]
train_x = train_x.drop(remove0, axis=1)
test_x = test_x.drop(remove1, axis=1)


# sort the columns in the same order
test_x = test_x[train_x.columns]

# scale the data set
train_x = MinMaxScaler().fit_transform(train_x)
test_x = MinMaxScaler().fit_transform(test_x)

# # print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

# use sklearn model to fit data and predict data
# model = linear_model.LinearRegression()
# model = SVR(kernel="linear")
# model = GradientBoostingRegressor()
# model = DecisionTreeRegressor()
model = Lasso()
model.fit(train_x, train_y)
predict_y = model.predict(test_x)
# print(predict_y)

# calculate msr and pearson correlation coefficient
msr = mean_squared_error(test_y, predict_y)
correlation = pearsonr(test_y, predict_y)[0]

# write data to part1 summary csv file
movie_id = list(data1["movie_id"])
content = [["z5240523", msr, correlation]]
df = pd.DataFrame(content, columns=["zid", "MSR", "correlation"])
df.to_csv("z5240523.PART1.summary.csv", index=False)

# write data to part1 output csv file

df = pd.DataFrame([movie_id, list(predict_y)])
df = df.T
df.columns = ["movie_id", "predicted_revenue"]
df = df.sort_values(by="movie_id", axis=0, ascending=True)
df.to_csv("z5240523.PART1.output.csv", index=False)

# print(msr)
# print(correlation)



# part 2

# y_value is rating column
train_y = data0["rating"]
test_y = data1["rating"]


# drop the text columns
train_x = data0.drop(["movie_id", "cast", "crew", "genres", "homepage", "keywords", "original_language", "original_title",
                      "overview", "production_companies", "production_countries", "release_date", "spoken_languages",
                      "status", "tagline", "revenue"], axis=1)
# one-hot
# use the correlation to choose columns
corr = train_x.corr()
test = corr["rating"].sort_values()
result = np.where((test < -0.1) | (test > 0.1))
column_name = list(test.index)
keep_columns = [column_name[i] for i in list(result[0])]

# keep the columns which has the correlation > 0.1, <-0.1 with revenue
train_x = data0[keep_columns]

# drop the y column
train_x = train_x.drop(["rating"], axis=1)

# for test x, find columns which both appear in train data frame and test data frame
test_x = data1.copy()
train_column0 = list(train_x)
train_column1 = list(test_x)
train_column = [i for i in train_column0 if i in train_column1]
remove0 = [i for i in train_column0 if i not in train_column]
remove1 = [i for i in train_column1 if i not in train_column]
train_x = train_x.drop(remove0, axis=1)
test_x = test_x.drop(remove1, axis=1)


# sort the columns in the same order
test_x = test_x[train_x.columns]

# scale the data set
train_x = MinMaxScaler().fit_transform(train_x)
test_x = MinMaxScaler().fit_transform(test_x)

# use model fit and predict data
# cla = DecisionTreeClassifier()
# cla = GradientBoostingClassifier()
# cla = RandomForestClassifier(10)
# cla.fit(train_x, train_y)
# predict_y = cla.predict(test_x)

p_list = []
r_list = []
a_list = []
for k in range(3, 30):
    cla = KNeighborsClassifier(k)
    cla.fit(train_x, train_y)
    predict_y = cla.predict(test_x)

    # calculate the average precision and recall
    average_precision = 0
    average_recall = 0

    for label in set(predict_y):
        average_precision += precision_score(test_y, predict_y, average="macro", labels=[label])

    for label in set(test_y):
        average_recall += recall_score(test_y, predict_y, average="macro", labels=[label])
    average_precision = average_precision / len(set(predict_y))
    average_recall = average_recall / len(set(test_y))
    acc = accuracy_score(test_y, predict_y)
    p_list.append(average_precision)
    r_list.append(average_recall)
    a_list.append(acc)
acc = max(a_list)
average_precision = p_list[a_list.index(acc)]
average_recall = r_list[a_list.index(acc)]
k = a_list.index(acc) + 3

# write data to part2 summary csv file
movie_id = list(data1["movie_id"])
content = [["z5240523", average_precision, average_recall, acc]]
df = pd.DataFrame(content, columns=["zid", "average_precision", "average_recall", "accuracy"])
df.to_csv("z5240523.PART2.summary.csv", index=False)

# write data to part2 output csv file
df = pd.DataFrame([movie_id, list(predict_y)])
df = df.T
df.columns = ["movie_id", "predicted_rating"]
df = df.sort_values(by="movie_id", axis=0, ascending=True)
df.to_csv("z5240523.PART2.output.csv", index=False)


# print(average_precision)
# print(average_recall)
# print(acc)

# print("finish")
