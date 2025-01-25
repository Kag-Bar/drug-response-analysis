import argparse
from FeatureExtractor import FeatureExtractor
from ModelTrainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Drug Response Analysis")
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Path to the configuration JSON file"
    )
    parser.add_argument(
        "--save_models",
        default=True,
        action="store_true",
        help="Flag to save trained models"
    )
    parser.add_argument(
        "--explore_data",
        default=True,
        action="store_true",
        help="Flag to explore data and train models"
    )
    args = parser.parse_args()

    if args.explore_data:
        feature_extractor = FeatureExtractor(args.cfg_path)
        model_trainer = ModelTrainer(feature_extractor)

        y_cols = feature_extractor.y_cols()
        if len(feature_extractor.cfg.get("chosen_features", [])) > 0:
            feature_cols = feature_extractor.cfg["chosen_features"]
        else:
            feature_cols = feature_extractor.extract_features()

        pca_flag = feature_extractor.cfg.get("pca_features", {}).get("pca", False)
        if pca_flag:
            pca, n_features, explained_var = feature_extractor.extract_features_pca()
            model_trainer.define_pca(pca, n_features)

        results = {"features": feature_cols}
        for y in y_cols:
            cv_splits = feature_extractor.cross_val_split(y_col=y, feature_cols=feature_cols)
            results[y] = model_trainer.train(cv_splits, y_col=y)

        if pca_flag:
            results_pca = {}
            for y in y_cols:
                cv_splits = feature_extractor.cross_val_split(y_col=y)
                results_pca[y] = model_trainer.train(cv_splits, y_col=y, pca=True)

        if len(y_cols) > 1:
            model_trainer.plot_agumentations_impact(y_cols, results)
            if pca_flag:
                model_trainer.plot_agumentations_impact(y_cols, results_pca, pca=True)

        print("Finished training exploration")

        if args.save_models:
            X_train, X_test, y_train, y_test = feature_extractor.train_test_split(y_col='y',
                                                                                  feature_cols=feature_cols,
                                                                                  test_size=0.15)
            for model_name in feature_extractor.cfg['models_to_save']:
                model_trainer.train_and_save_model(X_train, y_train, model_name, pca=pca_flag, save_model=True)

if __name__ == '__main__':
    main()