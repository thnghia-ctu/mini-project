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

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# CÁC HÀM CẦN THIẾT

def export_pdf(results):
    # Đăng ký font TTF hỗ trợ Unicode tiếng Việt
    pdfmetrics.registerFont(TTFont('Times', 'times.ttf'))  # hoặc 'arial.ttf'

    doc = SimpleDocTemplate("results/report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    # Áp dụng font 'Times' cho tất cả các style trong stylesheets
    for style_name in styles.byName:
        styles[style_name].fontName = 'Times'

    elements = []

    #  Tiêu đề
    elements.append(Paragraph("<b>PHÂN TÍCH & DỰ ĐOÁN DỮ LIỆU IRIS</b>", styles['Title']))
    elements.append(Spacer(1, 12))

     # Biểu đồ trực quan hóa dataset
    scatter_desc = """
        Để có cái nhìn trực quan về tập dữ liệu, chúng ta cần hiển thị dữ liệu. Có rất nhiều phương pháp trực quan hóa
        dữ liệu, tuy nhiên, đối với tập Iris, tôi dùng sơ đồ phân tán để hiển thị dữ liệu
        <br/><br/>
        Biểu đồ phân tán (Scatter Plot) thể hiện mối quan hệ giữa các cặp đặc trưng trong tập dữ liệu <b>Iris</b>.
        Mỗi điểm dữ liệu đại diện cho một bông hoa, được tô màu theo <b>loài (species)</b>.        
    """
    elements.append(Paragraph(scatter_desc, styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Image("results/chart.png", width=400, height=300))
    elements.append(Spacer(1, 12))

    # Bảng kết quả
    desc = "Báo cáo này trình bày kết quả huấn luyện và so sánh 5 mô hình Machine Learning trên tập dữ liệu IRIS."
    elements.append(Paragraph(desc, styles['Normal']))
    elements.append(Spacer(1, 12))
   
    table_data = [["Mô hình", "Độ chính xác"]] + [[name, round(info["accuracy"], 3)] for name, info in results.items()]
    table = Table(table_data)
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Biểu đồ
    elements.append(Image("results/confusion_matrix.png", width=400, height=300))
    elements.append(Spacer(1, 12))

    # 5️⃣ Kết luận
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    conclusion = f"Mô hình có hiệu suất cao nhất là: <b>{best_model}</b>."
    elements.append(Paragraph(conclusion, styles['Normal']))

    # Lưu PDF
    doc.build(elements)

# CODE CHÍNH
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
st.dataframe(train)

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
elif x_var == "All" and y_var != "All":
    st.subheader(f"So sánh tất cả X với '{y_var}'")
    fig, axes=plt.subplots(len_cols-1, 1, figsize=(4, 15))
    index_ax=0
    for i, col in enumerate(features):
        if col!=y_var:
            sns.scatterplot(data=train, x=col, y=y_var, hue='species', ax=axes[index_ax])
            index_ax=index_ax+1

elif y_var == "All" and x_var != "All":
    st.subheader(f"So sánh '{x_var}' với tất cả Y")
    fig, axes = plt.subplots(1, len_cols-1, figsize=(15, 4))
    index_ax=0
    for i, col in enumerate(features):
        if col!=x_var:
            sns.scatterplot(data=train, x=x_var, y=col, hue='species', ax=axes[index_ax])
            index_ax=index_ax+1

else:
    st.subheader(f"Biểu đồ tán xạ: {x_var} vs {y_var}")
    fig, ax = plt.subplots()
    sns.scatterplot(data=train, x=x_var, y=y_var, hue='species',  palette='Set2', ax=ax)

# === Lưu biểu đồ ra file PNG ===
chart_path = f"results/chart.png"
plt.savefig(chart_path, bbox_inches="tight")  # Lưu ảnh
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

# === Lưu biểu đồ ra file PNG ===
chart_path = "results/confusion_matrix.png"
plt.savefig(chart_path, bbox_inches="tight")  # Lưu ảnh

# Hiển thị ma trận nhằm lẫn
st.pyplot(fig)

# Tạo nút bấm xuất pdf
btn = st.button("📄 Xuất PDF", key="export_pdf")
st.markdown("""
    <style>
    .st-key-export_pdf {
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 999;
        button {
            background-color: #9ecfd4
        }
    }
    </style>
""", unsafe_allow_html=True)

if btn:
    export_pdf(results)
    st.success("✅ Đã tạo file PDF thành công!")
