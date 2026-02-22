import pandas as pd
import os
import sys
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_data, get_tfidf_vectorizer, get_binary_labels, get_feature_pipeline, combine_features

def run_stacking():
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
    
    # 4. Initialize Base Learners
    print("Initializing Base Learners...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)), # probability=True needed for stacking sometimes, or at least helpful
        ('xgb', XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lr', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ]
    
    # 5. Initialize Stacking Classifier
    print("Initializing Stacking Classifier (Meta-Learner: Logistic Regression)...")
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        n_jobs=-1,
        passthrough=False # False: Meta-learner only sees predictions of base learners
    )
    
    # 6. Train
    print("Training Stacking Ensemble (this will take a while as it trains all base models)...")
    clf.fit(X_train, y_train)
    
    # 7. Evaluate
    print("Evaluating on Validation Set...")
    y_pred = clf.predict(X_valid)
    
    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_valid, y_pred, target_names=['Fake', 'True']))

if __name__ == "__main__":
    run_stacking()
