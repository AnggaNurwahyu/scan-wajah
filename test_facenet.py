import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from keras_facenet import FaceNet



# 1. Load Dataset LFW


print("Loading LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
X, y = lfw_people.images, lfw_people.target
target_names = lfw_people.target_names

print("Dataset shape:", X.shape)
print("Total classes:", len(target_names))


# 2. Extract Embeddings with FaceNet

embedder = FaceNet()

# Mengubah grayscale (1 channel) jadi RGB (3 channel)
X_rgb = np.stack([X, X, X], axis=-1)

# Mengambil sebagian data biar cepat
X_small = X_rgb[:500]
y_small = y[:500]

print("Generating embeddings...")
embeddings = embedder.embeddings(X_small)


# 3. Split data untuk training & testing

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y_small, test_size=0.2, random_state=42
)


# 4. Train classifier (SVM)

print("Training classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)


# 5. Evaluasi

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))


# 6. Visualisasi contoh prediksi

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_small[i].astype('uint8'))
    pred_name = target_names[clf.predict([embeddings[i]])[0]]
    true_name = target_names[y_small[i]]
    ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
