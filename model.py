import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tqdm import tqdm
import time

print("Loading dataset...")
data = pd.read_csv("gestures.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Progress bar for pipeline steps
steps = ["Train-Test Split", "Model Training", "Evaluation", "Saving Model"]

with tqdm(total=len(steps), desc="Training Pipeline") as pbar:

    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    time.sleep(0.3)  # visual clarity for workshop
    pbar.update(1)

    # Step 2: Train model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    time.sleep(0.3)
    pbar.update(1)

    # Step 3: Evaluate
    accuracy = model.score(X_test, y_test)
    pbar.update(1)

    # Step 4: Save model
    joblib.dump(model, "model-2.pkl")
    pbar.update(1)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("Model saved as model.pkl")
