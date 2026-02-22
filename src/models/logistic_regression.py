import os
import sys
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_data, get_tfidf_vectorizer, get_binary_labels, get_feature_pipeline, combine_features

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def run_logistic_regression():
    train_df, valid_df, test_df = load_data()

    y_train = get_binary_labels(train_df["label"])
    y_valid = get_binary_labels(valid_df["label"])
    y_test = get_binary_labels(test_df["label"])

    vectorizer = get_tfidf_vectorizer(train_df["statement"])
    X_train_text = vectorizer.transform(train_df["statement"])
    X_valid_text = vectorizer.transform(valid_df["statement"])
    X_test_text = vectorizer.transform(test_df["statement"])

    metadata_pipeline = get_feature_pipeline()
    X_train_meta = metadata_pipeline.fit_transform(train_df)
    X_valid_meta = metadata_pipeline.transform(valid_df)
    X_test_meta = metadata_pipeline.transform(test_df)

    X_train = combine_features(X_train_text, X_train_meta)
    X_valid = combine_features(X_valid_text, X_valid_meta)
    X_test = combine_features(X_test_text, X_test_meta)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=class_weight_dict,
        C=1.0
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Fake", "True"]))


if __name__ == "__main__":
    run_logistic_regression()