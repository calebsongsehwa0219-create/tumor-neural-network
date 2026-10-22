import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# 교육용 공개 데이터 불러오기
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target  # 0: malignant(악성), 1: benign(양성)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 은닉층 구조가 다른 인공신경망 모델 비교
structures = {
    "1 hidden layer (8 neurons)": (8,),
    "1 hidden layer (16 neurons)": (16,),
    "2 hidden layers (16, 8 neurons)": (16, 8),
}

results = []

for name, structure in structures.items():
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=structure,
            max_iter=2000,
            random_state=42
        )
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    malignant_recall = recall_score(y_test, predictions, pos_label=0)

    results.append({
        "모델": name,
        "정확도": accuracy,
        "악성 재현율": malignant_recall
    })

    print(f"\n모델: {name}")
    print(f"정확도: {accuracy * 100:.2f}%")
    print(f"악성 재현율: {malignant_recall * 100:.2f}%")
    print("혼동 행렬 (실제 행, 예측 열; 순서: 악성, 양성)")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

df = pd.DataFrame(results)
print("\n===== 모델별 성능 비교 =====")
print(df.to_string(index=False))

ax = df.plot(
    x="모델",
    y=["정확도", "악성 재현율"],
    kind="bar",
    ylim=(0, 1.05),
    figsize=(9, 5)
)
ax.set_ylabel("Score (0 to 1)")
ax.set_title("Classification Performance by Neural Network Structure")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
