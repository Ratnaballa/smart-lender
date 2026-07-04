from src.ml_pipeline import train_and_save_model


if __name__ == "__main__":
    report = train_and_save_model()
    print("Best model:", report["best_model"])
    for name, metrics in report["results"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_score']:.3f}")
