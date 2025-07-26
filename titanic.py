import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('E:/study/PycharmProjects/KaggleProject01/.venv/dataset/train.csv')
test = pd.read_csv('E:/study/PycharmProjects/KaggleProject01/.venv/dataset/test.csv')

# 查看数据概况
print(train.info())
print(train.describe())

# 生存率可视化
sns.barplot(x='Pclass', y='Survived', data=train)  # 舱位等级
sns.barplot(x='Sex', y='Survived', data=train)     # 性别
plt.show()

# 合并数据集
all_data = pd.concat([train, test], sort=False)

# 提取称呼特征
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# 填充缺失值
all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S')

# 创建新特征
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
all_data['Deck'] = all_data['Cabin'].str[0]  # 提取船舱甲板信息

# 分类数据编码
all_data = pd.get_dummies(all_data, columns=['Sex', 'Embarked', 'Title', 'Pclass', 'Deck'], drop_first=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 分割数据集
train_clean = all_data[~all_data['Survived'].isna()]
test_clean = all_data[all_data['Survived'].isna()]

X = train_clean.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train_clean['Survived']

# 参数调优
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

model = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=5, verbose=1)
model.fit(X, y)
print(f"Best parameters: {model.best_params_}")

test_final = test_clean.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
predictions = model.predict(test_final).astype(int)

submission = pd.DataFrame({
    'PassengerId': test_clean['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)