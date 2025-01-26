import joblib
import argparse
from InferenceFeatureExtractor import InferenceFeatureExtractor
from ModelTest import ModelTest

def main():
    parser = argparse.ArgumentParser(description="Drug Response Analysis")
    parser.add_argument(
        "--cfg_path",
        default='cfg_test.json',
        type=str,
        help="Path to the training configuration JSON file"
    )
    args = parser.parse_args()

    feature_extractor = InferenceFeatureExtractor(args.cfg_path)
    X = feature_extractor.prepare_data()
    model_tester = ModelTest(feature_extractor)
    pca_flag = feature_extractor.cfg.get("use_pca", False)
    if pca_flag:
        pca = joblib.load(feature_extractor.cfg['pca']['pca_path'])
        n_features = feature_extractor.cfg['pca']['n_features']
        model_tester.define_pca(pca, n_features)

    for model_name, model_path in feature_extractor.cfg['models_path'].items():
        model_tester.load_model(model_name, model_path)
        model_tester.run_model(X, pca_flag)
        print("Finished testing!")

if __name__ == '__main__':
    main()