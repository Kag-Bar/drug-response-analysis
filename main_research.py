import argparse
from FeatureExtractor import FeatureExtractor
from ModelTrainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Drug Response Analysis")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default='cfg.json',
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    feature_extractor = FeatureExtractor(args.cfg_path)
    model_trainer = ModelTrainer(feature_extractor)

    y_cols = feature_extractor.y_cols()
    if len(feature_extractor.cfg.get("chosen_features", [])) > 0:
        feature_cols = feature_extractor.cfg["chosen_features"]
    else:
        feature_cols = feature_extractor.extract_features()

    results = {"features": feature_cols}
    for y in y_cols:
        cv_splits = feature_extractor.cross_val_split(y_col=y, feature_cols=feature_cols)
        results[y] = model_trainer.train(cv_splits, y_col=y)
    if len(y_cols) > 1:
        model_trainer.plot_agumentations_impact(y_cols, results)

    pca_flag = feature_extractor.cfg.get("pca_features", {}).get("pca", False)
    if pca_flag:
        pca, n_features, explained_var = feature_extractor.extract_features_pca()
        model_trainer.define_pca(pca, n_features)

        results_pca = {}
        for y in y_cols:
            cv_splits = feature_extractor.cross_val_split(y_col=y)
            results_pca[y] = model_trainer.train(cv_splits, y_col=y, pca=True)
        if len(y_cols) > 1:
            model_trainer.plot_agumentations_impact(y_cols, results_pca, pca=True)

    print("Finished training exploration")

if __name__ == '__main__':
    main()