import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_data, get_tfidf_vectorizer, get_binary_labels, get_feature_pipeline, combine_features

def run_random_forest():
    # 1. Load Data
    print("Loading data...")
    train_df, valid_df, test_df = load_data()

    # 2. Get Binary Labels
    y_train = get_binary_labels(train_df['label'])
    y_valid = get_binary_labels(valid_df['label'])
    
    # 3. Features
    print("Extracting features (Text + Metadata)...")
    
    # Text Features (TF-IDF)
    print(" - Vectorizing text...")
    vectorizer = get_tfidf_vectorizer(train_df['statement'])
    X_train_text = vectorizer.transform(train_df['statement'])
    X_valid_text = vectorizer.transform(valid_df['statement'])
    
    # Metadata Features
    print(" - Processing metadata...")
    metadata_pipeline = get_feature_pipeline()
    X_train_meta = metadata_pipeline.fit_transform(train_df)
    X_valid_meta = metadata_pipeline.transform(valid_df)
    
    # Combine
    print(" - Combining features...")
    X_train = combine_features(X_train_text, X_train_meta)
    X_valid = combine_features(X_valid_text, X_valid_meta)
    
    # 4. Initialize Random Forest
    print("Initializing Random Forest (Binary Classification)...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # 5. Train
    print("Training Random Forest model...")
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    print("Evaluating on Validation Set...")
    y_pred = clf.predict(X_valid)
    
    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_valid, y_pred, target_names=['Fake', 'True']))

if __name__ == "__main__":
    run_random_forest()
