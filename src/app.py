import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        HỆ THỐNG PHÂN TÍCH VÀ DỰ ĐOÁN TRÊN DỮ LIỆU IRIS
    </h1>
    """,
    unsafe_allow_html=True
)

st.title("Hiển thị bảng dữ liệu")

# Đọc dữ liệu train và test
train = pd.read_csv("data\iris\iris.trn", header=None)
test  = pd.read_csv("data\iris\iris.tst", header=None)

# Đặt tên cột
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
features= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
label='species'
train.columns = columns
test.columns = columns

# Chia đặc trưng và nhãn
X_train = train[features]
y_train = train[label]

X_test = test[features]
y_test = test[label]

# Hiển thị bảng
st.dataframe(train.head())

st.title("Phân tích dữ liệu Iris")
# Chọn biến X và Y
col1, col2 = st.columns(2)
x_var = col1.selectbox("Chọn trục X:", ['All']+features, index=0)
y_var = col2.selectbox("Chọn trục Y:", ['All']+features, index=0)

len_cols=len(features)
if x_var == "All" and y_var == "All":
    fig =sns.pairplot(
        train,
        hue='species',       # tô màu theo loài hoa
        palette='Set2',      # bảng màu
        diag_kind='kde',     # biểu đồ mật độ ở đường chéo (có thể đổi 'hist')
        corner=True          # nếu True thì chỉ vẽ nửa dưới
    )
    st.pyplot(fig)
elif x_var == "All" and y_var != "All":
    st.subheader(f"So sánh tất cả X với '{y_var}'")
    fig, axes=plt.subplots(len_cols-1, 1, figsize=(4, 15))
    index_ax=0
    for i, col in enumerate(features):
        if col!=y_var:
            sns.scatterplot(data=train, x=col, y=y_var, hue='species', ax=axes[index_ax])
            index_ax=index_ax+1
    st.pyplot(fig)

elif y_var == "All" and x_var != "All":
    st.subheader(f"So sánh '{x_var}' với tất Y")
    fig, axes = plt.subplots(1, len_cols, figsize=(15, 4))
    index_ax=0
    for i, col in enumerate(features):
        if col!=x_var:
            sns.scatterplot(data=train, x=x_var, y=col, hue='species', ax=axes[index_ax])
            index_ax=index_ax+1
    st.pyplot(fig)

else:
    st.subheader(f"Biểu đồ tán xạ: {x_var} vs {y_var}")
    fig, ax = plt.subplots()
    sns.scatterplot(data=train, x=x_var, y=y_var, hue='species',  palette='Set2', ax=ax)
    st.pyplot(fig)


# Danh sách mô hình
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN":KNeighborsClassifier(n_neighbors=3),
    "NB":GaussianNB(),
    "SVM":SVC()
}

if "results" not in st.session_state:
    st.session_state.results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.session_state.results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_test, y_pred)
        }

results = st.session_state.results

st.title("Huấn luyện mô hình")
mol=st.selectbox("Chọn mô hình huấn luyện: ", models.keys())


# Hiển thị độ chính xác
st.write(f"**Độ chính xác (Accuracy)**: {results[mol]['accuracy']:.3f}")

report_dict = classification_report(y_test, results[mol]['y_pred'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
st.dataframe(report_df)

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, results[mol]['y_pred'])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
ax.set_title(f"Ma trận nhầm lẫn: {mol}")
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
st.pyplot(fig)