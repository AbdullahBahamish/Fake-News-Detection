import argparse
from models.svm import run_svm
from models.logistic_regression import run_logistic_regression
from models.random_forest import run_random_forest
from models.xgboost import run_xgboost
from models.bert_baseline import run_bert
from models.stacking import run_stacking

def main():
    parser = argparse.ArgumentParser(description='Run Fake News Detection Models (Binary Classification: Fake vs True)')
    parser.add_argument('--model', type=str, choices=['svm', 'lr', 'rf', 'xgb', 'bert', 'stacking', 'all'], default='all',
                        help='Model to run: svm, lr, rf, xgb, bert, stacking, or all')
    
    args = parser.parse_args()

    if args.model in ['lr', 'all']:
        print("\n" + "="*50)
        print("Running Logistic Regression...")
        print("="*50)
        run_logistic_regression()

    if args.model in ['svm', 'all']:
        print("\n" + "="*50)
        print("Running SVM...")
        print("="*50)
        run_svm()

    if args.model in ['rf', 'all']:
        print("\n" + "="*50)
        print("Running Random Forest...")
        print("="*50)
        run_random_forest()

    if args.model in ['xgb', 'all']:
        print("\n" + "="*50)
        print("Running XGBoost...")
        print("="*50)
        run_xgboost()

    if args.model in ['stacking', 'all']:
        print("\n" + "="*50)
        print("Running Stacking Ensemble...")
        print("="*50)
        run_stacking()

    if args.model in ['bert', 'all']:
        print("\n" + "="*50)
        print("Running BERT Baseline...")
        print("="*50)
        run_bert()

if __name__ == "__main__":
    main()
