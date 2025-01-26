import joblib
import argparse
from InferenceFeatureExtractor import InferenceFeatureExtractor
from ModelTrainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Drug Response Analysis")
    parser.add_argument(
        "--cfg_path",
        default='cfg_train.json',
        type=str,
        help="Path to the training configuration JSON file"
    )
    args = parser.parse_args()

    feature_extractor = InferenceFeatureExtractor(args.cfg_path)
    X_train, X_test, y_train, y_test = feature_extractor.prepare_data("training")
    model_trainer = ModelTrainer(feature_extractor)
    pca_flag = feature_extractor.cfg.get("use_pca", False)
    if pca_flag:
        pca = joblib.load(feature_extractor.cfg['pca']['pca_path'])
        n_features = feature_extractor.cfg['pca']['n_features']
        model_trainer.define_pca(pca, n_features)

    for model_name in feature_extractor.cfg['models_to_save']:
        model = model_trainer.train_and_save_model(X_train, y_train, model_name, pca=pca_flag, save_model=True)
        model_trainer.validate_model(model, X_test, y_test, model_name, pca=pca_flag)

if __name__ == '__main__':
    main()