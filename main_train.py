
from FeatureExtractor import FeatureExtractor
from ModelTrainer import ModelTrainer

def main():
    cfg_path = 'E:\Studies\pythonProject\drug-response-analysis\cfg.json' #### take as an argument
    feature_extractor = FeatureExtractor(cfg_path)
    model_trainer = ModelTrainer() #### BUILD

    y_cols = feature_extractor.y_cols()
    feature_cols = feature_extractor.extract_features()
    pca, n_features, explained_var = feature_extractor.extract_features_pca()

    model_trainer.define_pca(pca, n_features)

    results = {}
    for y in y_cols:
        cv_splits = feature_extractor.cross_val_split(y_col=y, feature_cols=feature_cols)
        results[y] = model_trainer.train(cv_splits, y_col=y)
    for y in y_cols:
        cv_splits = feature_extractor.cross_val_split(y_col=y)
        results_pca = model_trainer.train(cv_splits, y_col=y, pca=True)



if __name__ == '__main__':
    main()