import traceback

from src.train import main as train_main

def debug():
    try:
        print("Starting diagnostic run of Stacking model...")
        train_main(["--model", "stacking"])
    except Exception as e:
        print("\n" + "="*50)
        print("CAUGHT EXCEPTION:")
        print("="*50)
        traceback.print_exc()

if __name__ == "__main__":
    debug()
