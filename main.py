from omegaconf import OmegaConf

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.evaluation import evaluate_model, evaluate_catboost, evaluate_dnn, train_test_final_dnn

from src.ensembles import get_oof_lgbm, get_oof_catboost, get_oof_dnn

def main():
    
    config = OmegaConf.load('config/config.yaml')
    
    train = pd.read_csv(config.paths.train_path)
        
    target = config.target.column

    X = train.drop(columns=target)
    y = train[target]
    
    if config.target.log_transform:
        y = np.log1p(y)
        
    results = []
    
    for model_name, model_cfg in config.models.items():
        if not model_cfg.enabled:
            continue
        
        print(f'running: {model_name}')
        
        result = evaluate_configured_model(
            model_name=model_name,
            model_cfg=model_cfg,
            X=X,
            y=y,
            config=config
        )
        
        results.append({
            "model": model_name,
            "cv_mean": result["mean_rmse"],
            "cv_std": result["std_rmse"],
        })
        
    if not results:
        raise ValueError("No models are enabled in config.")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values(
        "cv_mean",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        config.paths.results_path,
        index=False,
    )
    
    print('\nResults:')
    print(results_df)
    
    best_model_name = results_df.iloc[0]["model"]
    best_score = results_df.iloc[0]["cv_mean"]

    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_score:.4f}")
    
    test = pd.read_csv(config.paths.test_path)
    
    if config.general.use_ensemble_for_submission:
        print("\nTraining ensemble and generating submission...")
        train_ensemble_and_submit(
            X=X,
            y=y,
            test=test,
            config=config
        )
        
    else:
        print(f"\nTraining best model ({best_model_name}) and generating submission...")
        best_model_cfg = config.models[best_model_name]
        
        train_best_and_submit(
            X=X,
            y=y,
            test=test,
            best_model_name=best_model_name,
            best_model_cfg=best_model_cfg,
            config=config
        )

        
def evaluate_configured_model(model_name, model_cfg, X, y, config):
    
    model_params = OmegaConf.to_container(model_cfg.params, resolve=True)
    
    if model_name == 'dnn':
        result = evaluate_dnn(
            X=X,
            y=y,
            model_params=model_params,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            epochs=model_cfg.training.epochs,
            batch_size=model_cfg.training.batch_size,
            lr=model_cfg.training.lr,
            weight_decay=model_cfg.training.weight_decay,
            
            use_house_age=config.preprocessing.use_house_age,
            use_total_bathrooms=config.preprocessing.use_total_bathrooms,
            use_total_sf=config.preprocessing.use_total_sf,
        )
        
        return result
    
    if model_name == 'catboost':
        result = evaluate_catboost(
            X=X,
            y=y,
            model_params=model_params,
            n_splits=config.cv.n_splits,
            seed=config.general.seed,
            early_stopping_rounds=model_cfg.training.early_stopping_rounds,
            
            use_house_age=config.preprocessing.use_house_age,
            use_total_bathrooms=config.preprocessing.use_total_bathrooms,
            use_total_sf=config.preprocessing.use_total_sf,
        )
        
        return result
    
    preprocessor = build_preprocessor(
        X=X,
        model_type=model_cfg.model_type, 
        use_total_sf=config.preprocessing.use_total_sf, 
        use_total_bathrooms=config.preprocessing.use_total_bathrooms, 
        use_house_age=config.preprocessing.use_house_age
        )
    
    model = get_model(
        model_name=model_name,
        seed=config.general.seed,
        **model_params
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return evaluate_model(
        estimator=pipeline,
        X=X,
        y=y,
        n_splits=config.cv.n_splits,
        scoring=config.cv.scoring,
        seed=config.general.seed,
    )
      
def train_ensemble_and_submit(X, y, test, config):
    seed = config.general.seed
    n_splits = config.cv.n_splits
    
    catboost_params = OmegaConf.to_container(
        config.models.catboost.params,
        resolve=True,
    )
    
    lgbm_params = OmegaConf.to_container(
        config.models.lightgbm.params,
        resolve=True,
    )
    
    dnn_params = OmegaConf.to_container(
        config.models.dnn.params,
        resolve=True,
    )
    
    cat_oof, cat_test = get_oof_catboost(
        X=X,
        y=y,
        test=test,
        model_params=catboost_params,
        n_splits=n_splits,
        seed=seed,
        
        use_house_age=config.preprocessing.use_house_age,
        use_total_bathrooms=config.preprocessing.use_total_bathrooms,
        use_total_sf=config.preprocessing.use_total_sf,
    )
    
    lgbm_oof, lgbm_test = get_oof_lgbm(
        X=X,
        y=y,
        test=test,
        model_params=lgbm_params,
        n_splits=n_splits,
        seed=seed,
        
        use_house_age=config.preprocessing.use_house_age,
        use_total_bathrooms=config.preprocessing.use_total_bathrooms,
        use_total_sf=config.preprocessing.use_total_sf,
    )
    
    dnn_oof, dnn_test = get_oof_dnn(
        X=X,
        y=y,
        test=test,
        hidden_dim=dnn_params['hidden_dim'],
        epochs=config.models.dnn.training.epochs,
        batch_size=config.models.dnn.training.batch_size,
        lr=config.models.dnn.training.lr,
        weight_decay=config.models.dnn.training.weight_decay,
        n_splits=n_splits,
        seed=seed,
        
        use_house_age=config.preprocessing.use_house_age,
        use_total_bathrooms=config.preprocessing.use_total_bathrooms,
        use_total_sf=config.preprocessing.use_total_sf,
    )
    
    ensemble_oof = (
        0.6 * cat_oof +
        0.3 * lgbm_oof +
        0.1 * dnn_oof
    )
    
    ensemble_test = (
        0.6 * cat_test +
        0.3 * lgbm_test +
        0.1 * dnn_test
    )
    
    ensemble_rmse = np.sqrt(np.mean((y.to_numpy() - ensemble_oof) ** 2))
    
    print(
        f"\nEnsemble OOF RMSE: "
        f"{ensemble_rmse:.6f}"
    )
    
    if config.target.log_transform:
        ensemble_test = np.expm1(ensemble_test)
        
    save_submission(
        test=test,
        preds=ensemble_test,
        path=config.paths.ensemble_submission_path,
    )
    

def train_best_and_submit(X, y, test, best_model_name,
                          best_model_cfg, config):
    
    if best_model_name == 'catboost':
        train_catboost_and_submit(
            X=X,
            y=y,
            test=test,
            model_cfg=best_model_cfg,
            config=config
        )
        return
    
    if best_model_name == 'dnn':
        train_dnn_and_submit(
            X=X,
            y=y,
            test=test,
            model_cfg=best_model_cfg,
            config=config,
        )
        return

    train_sklearn_and_submit(
        X=X,
        y=y,
        test=test,
        model_name=best_model_name,
        model_cfg=best_model_cfg,
        config=config,
    )   


def train_sklearn_and_submit(X, y, test, model_name,
                            model_cfg, config):
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    preprocessor = build_preprocessor(
        X, 
        model_cfg.model_type, 
        use_total_sf=config.preprocessing.use_total_sf, 
        use_total_bathrooms=config.preprocessing.use_total_bathrooms, 
        use_house_age=config.preprocessing.use_house_age
    )
    
    model = get_model(
        model_name=model_name,
        seed=config.general.seed,
        **model_params
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X, y)
    
    preds = pipeline.predict(test)
    
    if config.target.log_transform:
        preds = np.expm1(preds)
    
    save_submission(
        test=test, 
        preds=preds, 
        path=config.paths.submission_path
    )


def train_catboost_and_submit(X, y, test,
                            model_cfg, config):
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    preprocessor = build_preprocessor(
        X, 
        model_type='catboost', 
        use_total_sf=config.preprocessing.use_total_sf, 
        use_total_bathrooms=config.preprocessing.use_total_bathrooms, 
        use_house_age=config.preprocessing.use_house_age
    )
    
    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)
    
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    categorical_features.append("MSSubClass")

    n_cat_features = len(categorical_features)

    cat_features = list(range(n_cat_features))
    
    model = get_model(
        model_name='catboost',
        seed=config.general.seed,
        **model_params
    )

    model.fit(
        X_processed,
        y,
        cat_features=cat_features,
    )

    predictions = model.predict(test_processed)
    
    if config.target.log_transform:
        predictions = np.expm1(predictions)

    save_submission(
        test=test,
        preds=predictions,
        path=config.paths.submission_path,
    )


def train_dnn_and_submit(X, y, test,
                        model_cfg, config):
    
    model_params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True,
    )
    
    preprocessor = build_preprocessor(
        X, 
        model_cfg.model_type, 
        use_total_sf=config.preprocessing.use_total_sf, 
        use_total_bathrooms=config.preprocessing.use_total_bathrooms, 
        use_house_age=config.preprocessing.use_house_age
    )
    
    X_processed = preprocessor.fit_transform(X)
    test_processed = preprocessor.transform(test)
    
    preds = train_test_final_dnn(
        X_train=X_processed,
        y_train=y,
        X_test=test_processed,
        hidden_dim=model_params['hidden_dim'],
        epochs=model_cfg.training.epochs,
        batch_size=model_cfg.training.batch_size,
        lr=model_cfg.training.lr,
        weight_decay=model_cfg.training.weight_decay,
        log_transform=config.target.log_transform,
        seed=config.general.seed
    )
    
    save_submission(
        test=test,
        preds=preds,
        path=config.paths.submission_path,
    )
    
    
def save_submission(test, preds, path):
    submission = pd.DataFrame({
        'Id': test['Id'], 
        'SalePrice': preds
    })
    
    submission.to_csv(
        path,
        index=False
    )
    
    print('submission saved to', path)
    
if __name__ == "__main__":
    main()