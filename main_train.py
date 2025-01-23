
from FeatureExtractor import FeatureExtractor

def main():
    cfg_path = '' #### take as an argument
    feature_extractor = FeatureExtractor(cfg_path)
    model_trainer = ModelTrainer(cfg_path) #### BUILD


    feature_cols = feature_extractor.extract_features() ### either this or with a PCA
    y_cols = feature_extractor.y_cols()
    for y in y_cols:
        cv_splits = feature_extractor.cross_val_split(y_col=y, feature_cols=feature_cols)
        model_trainer.train(cv_splits) ### Build


if __name__ == '__main__':
    main()