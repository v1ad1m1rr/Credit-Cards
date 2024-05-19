import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

# Заголовок и описание проекта
st.title("Прогнозирование дефолта по кредитным картам")

st.markdown("""
# Прогнозирование дефолта по кредитным картам

- **Технология**: Машинное обучение
- **Сфера**: Банковское дело

## Постановка задачи
Финансовые угрозы демонстрируют тенденцию к увеличению кредитных рисков коммерческих банков. 
Одной из крупнейших угроз, с которой сталкиваются коммерческие банки, является предсказание вероятности дефолта клиентов. 
Цель данного проекта - предсказать вероятность дефолта на основе характеристик владельца кредитной карты и истории платежей.

## Результаты
Необходимо построить решение, которое будет предсказывать вероятность дефолта на основе характеристик владельца кредитной карты и истории платежей.
""")


# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("UCI_Credit_Card.csv")
    df.rename(columns={'PAY_0': 'PAY_1', 'default.payment.next.month': 'Default'}, inplace=True)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 6], 5)  # Объединение категорий образования
    df['AgeBin'] = pd.cut(df['AGE'], bins=[20, 25, 30, 35, 40, 50, 60, 80],
                          labels=['(20, 25]', '(25, 30]', '(30, 35]', '(35, 40]', '(40, 50]', '(50, 60]', '(60, 80]'])
    df['LimitBin'] = pd.cut(df['LIMIT_BAL'],
                            bins=[5000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1100000])
    return df


df = load_data()

# Заголовок
st.title("Анализ данных по кредитным картам клиентов")

# Визуализация данных
st.header("Распределение клиентов по полу и вероятности дефолта")
fig, ax = plt.subplots(figsize=(12, 4))
sns.countplot(data=df, x='SEX', hue='Default', palette='rocket', ax=ax)
plt.xlabel("Пол")
plt.ylabel("Количество клиентов")
plt.xticks([0, 1], ['Мужчины', 'Женщины'])
st.pyplot(fig)

st.header("Распределение клиентов по образованию и вероятности дефолта")
fig, ax = plt.subplots(figsize=(12, 4))
sns.countplot(data=df, x='EDUCATION', hue='Default', palette='rocket', ax=ax)
plt.xlabel("Образование")
plt.ylabel("Количество клиентов")
plt.xticks([0, 1, 2, 3, 4], ['Аспирантура', 'Университет', 'Средняя школа', 'Другое', 'Неизвестно'])
st.pyplot(fig)

st.header("Распределение клиентов по семейному положению и вероятности дефолта")
fig, ax = plt.subplots(figsize=(12, 4))
sns.countplot(data=df, x='MARRIAGE', hue='Default', palette='rocket', ax=ax)
plt.xlabel("Семейное положение")
plt.ylabel("Количество клиентов")
plt.xticks([0, 1, 2, 3], ['Неизвестно', 'Женат', 'Холост', 'Разведен'])
st.pyplot(fig)

st.header("Распределение клиентов по возрастным группам и вероятности дефолта")
fig, ax = plt.subplots(figsize=(12, 4))
sns.countplot(data=df, x='AgeBin', hue='Default', palette='rocket', ax=ax)
plt.xlabel("Возрастная группа")
plt.ylabel("Количество клиентов")
st.pyplot(fig)

st.header("Распределение клиентов по кредитному лимиту и вероятности дефолта")
fig, ax = plt.subplots(figsize=(14, 4))
sns.countplot(data=df, x='LimitBin', hue='Default', palette='rocket', ax=ax)
plt.xlabel("Кредитный лимит")
plt.ylabel("Количество клиентов")
st.pyplot(fig)

st.header("Влияние пола на кредитный лимит")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='SEX', y='LIMIT_BAL', data=df, palette='rocket', showmeans=True,
            meanprops={"markerfacecolor": "red", "markeredgecolor": "black", "markersize": "10"}, ax=ax)
plt.xlabel("Пол")
plt.ylabel("Кредитный лимит")
plt.xticks([0, 1], ['Мужчины', 'Женщины'])
st.pyplot(fig)

# Выбор дополнительных параметров для анализа
st.header("Выберите параметры для анализа")
feature = st.selectbox("Выберите характеристику", ['EDUCATION', 'MARRIAGE', 'AgeBin', 'LimitBin'])
hue = st.selectbox("Выберите цветовую кодировку", ['SEX', 'Default'])

st.header(f"Влияние {feature} на кредитный лимит")
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x=feature, y='LIMIT_BAL', hue=hue, data=df, palette='rocket', showmeans=True,
            meanprops={"markerfacecolor": "red", "markeredgecolor": "black", "markersize": "10"}, ax=ax)
plt.xlabel(feature)
plt.ylabel("Кредитный лимит")
st.pyplot(fig)

df = df[df.select_dtypes(include=['int64', 'float64']).columns.tolist()]

# # Очистка данных
X1 = df.copy().drop('Default', axis=1)

# y = df['Default']
# pay_x_new = ['PAY_1_new', 'PAY_2_new', 'PAY_3_new', 'PAY_4_new', 'PAY_5_new', 'PAY_6_new']
# bill_amtx_bins = ['BILL_AMT1_bin', 'BILL_AMT2_bin', 'BILL_AMT3_bin', 'BILL_AMT4_bin', 'BILL_AMT5_bin', 'BILL_AMT6_bin']
# pay_amtx_bins = ['PAY_AMT1_bin', 'PAY_AMT2_bin', 'PAY_AMT3_bin', 'PAY_AMT4_bin', 'PAY_AMT5_bin', 'PAY_AMT6_bin']
# X_base = X1.drop(columns=pay_x_new + bill_amtx_bins + pay_amtx_bins + ['AgeBin', 'LimitBin', 'ID'])

# Корреляция признаков

st.header("Корреляция признаков")

correlation = df.corr()
plt.figure(figsize=(30, 10))
sns.heatmap(correlation, square=True, annot=True, fmt=".1f")
st.pyplot(plt)

plt.figure(figsize=(20, 10))
X1.corrwith(df['Default']).plot.bar(title="Correlation with Default", fontsize=20, rot=90)
st.pyplot(plt)

X = df.drop(['ID', 'Default'], axis=1)
y = df['Default']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write("Размер тренировочной выборки:", X_train.shape)
st.write("Размер тестовой выборки:", X_test.shape)

# Логистическая регрессия
st.markdown("## Логистическая регрессия")
logmodel = LogisticRegression(random_state=1)
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)

roc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1, roc]],
                       columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC'])

st.write("Результаты логистической регрессии")
st.write(results)

# Вычисление средних значений
mean_values = df.mean()

# Форма для ввода данных
st.header("Введите данные для прогноза")

st.markdown("### Личная информация")
col1, col2 = st.columns(2)
with col1:
    LIMIT_BAL = st.number_input("Лимит по кредитной карте (LIMIT_BAL)", min_value=0, max_value=1000000,
                                value=int(mean_values['LIMIT_BAL']))
    SEX = st.selectbox("Пол (SEX)", options=[1, 2], format_func=lambda x: 'Мужской' if x == 1 else 'Женский',
                       index=0 if mean_values['SEX'] == 1 else 1)
    EDUCATION = st.selectbox("Образование (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda
        x: 'Высшая школа' if x == 1 else 'Университет' if x == 2 else 'Средняя школа' if x == 3 else 'Другое',
                             index=int(mean_values['EDUCATION']) - 1)
with col2:
    MARRIAGE = st.selectbox("Семейное положение (MARRIAGE)", options=[1, 2, 3], format_func=lambda
        x: 'Женат/Замужем' if x == 1 else 'Не женат/Не замужем' if x == 2 else 'Другое',
                            index=int(mean_values['MARRIAGE']) - 1)
    AGE = st.number_input("Возраст (AGE)", min_value=18, max_value=100, value=int(mean_values['AGE']))

st.markdown("### История платежей")
col3, col4, col5 = st.columns(3)
with col3:
    PAY_1 = st.number_input("Сентябрь (PAY_1)", min_value=-2, max_value=8, value=int(mean_values['PAY_1']))
    PAY_4 = st.number_input("Июнь (PAY_4)", min_value=-2, max_value=8, value=int(mean_values['PAY_4']))
    BILL_AMT1 = st.number_input("Сентябрь (BILL_AMT1)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT1']))
    BILL_AMT4 = st.number_input("Июнь (BILL_AMT4)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT4']))
    PAY_AMT1 = st.number_input("Сентябрь (PAY_AMT1)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT1']))
    PAY_AMT4 = st.number_input("Июнь (PAY_AMT4)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT4']))
with col4:
    PAY_2 = st.number_input("Август (PAY_2)", min_value=-2, max_value=8, value=int(mean_values['PAY_2']))
    PAY_5 = st.number_input("Май (PAY_5)", min_value=-2, max_value=8, value=int(mean_values['PAY_5']))
    BILL_AMT2 = st.number_input("Август (BILL_AMT2)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT2']))
    BILL_AMT5 = st.number_input("Май (BILL_AMT5)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT5']))
    PAY_AMT2 = st.number_input("Август (PAY_AMT2)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT2']))
    PAY_AMT5 = st.number_input("Май (PAY_AMT5)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT5']))
with col5:
    PAY_3 = st.number_input("Июль (PAY_3)", min_value=-2, max_value=8, value=int(mean_values['PAY_3']))
    PAY_6 = st.number_input("Апрель (PAY_6)", min_value=-2, max_value=8, value=int(mean_values['PAY_6']))
    BILL_AMT3 = st.number_input("Июль (BILL_AMT3)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT3']))
    BILL_AMT6 = st.number_input("Апрель (BILL_AMT6)", min_value=-500000, max_value=500000,
                                value=int(mean_values['BILL_AMT6']))
    PAY_AMT3 = st.number_input("Июль (PAY_AMT3)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT3']))
    PAY_AMT6 = st.number_input("Апрель (PAY_AMT6)", min_value=0, max_value=500000, value=int(mean_values['PAY_AMT6']))

# Формирование данных для прогноза
input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                        BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
                        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])

# Прогнозирование
if st.button("Прогнозировать", use_container_width=True, type='primary'):
    prediction = logmodel.predict(input_data)
    prediction_proba = logmodel.predict_proba(input_data)

    st.subheader("Результат прогнозирования")
    st.write("Вероятность дефолта:", prediction_proba[0][1])
    st.info("Клиент попадет в дефолт" if prediction[0] == 1 else "Клиент не попадет в дефолт")

    # Дополнительные графики
    st.markdown("## Дополнительные графики")

    # График вероятностей
    fig, ax = plt.subplots()
    ax.bar(['Не дефолт', 'Дефолт'], prediction_proba[0], color=['green', 'red'])
    ax.set_ylabel('Вероятность')
    st.pyplot(fig)
