import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


# Huấn luyện mô hình
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
ax.set_title("Ma trận nhầm lẫn (Confusion Matrix)")
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực tế")
st.pyplot(fig)
