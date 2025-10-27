import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu train và test
train = pd.read_csv("data\iris\iris.trn", header=None)
test  = pd.read_csv("data\iris\iris.tst", header=None)

# Đặt tên cột
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
train.columns = columns
test.columns = columns

# sns.pairplot(
#     train,
#     hue='species',       # tô màu theo loài hoa
#     palette='Set2',      # bảng màu
#     diag_kind='kde',     # biểu đồ mật độ ở đường chéo (có thể đổi 'hist')
#     corner=True          # nếu True thì chỉ vẽ nửa dưới
# )
# plt.show()

# Chia đặc trưng và nhãn
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']

X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Huấn luyện mô hình
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
print("Độ chính xác:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()