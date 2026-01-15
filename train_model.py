import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('training_data.csv')

# 2. Split into Features (X) and Target Class (y)
X = df.drop('class', axis=1) # Coordinates
y = df['class'] # Labels ("Normal", "Attack")

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# 4. Setup Pipeline (Standardize data -> Random Forest)
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

# 5. Train the Model
print("Training the brain...")
model = pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the Model
joblib.dump(model, 'harassment_model.pkl')
print("Model saved as 'harassment_model.pkl'!")