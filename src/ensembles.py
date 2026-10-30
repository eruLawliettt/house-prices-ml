import numpy as np
import torch

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

from src.preprocessing import build_preprocessor
from src.models import get_model
from src.evaluation import HousePricesMLP


def get_oof_lgbm(X, y, test, model_params, n_splits=5,
                 seed=42, use_total_sf=True,use_total_bathrooms=True, 
                 use_house_age=True):
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    
    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type='boosting',
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        model = get_model(
            'lightgbm',
            seed=seed,
            **model_params,
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        oof[val_idx] = pipeline.predict(X_val)
        
        test_preds += (
            pipeline.predict(test) / n_splits
        )
        
    return oof, test_preds


def get_oof_catboost(X, y, test, model_params, n_splits=5,
                    seed=42, use_total_sf=True,use_total_bathrooms=True, 
                    use_house_age=True):
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    
    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type='catboost',
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        test_processed = preprocessor.transform(test)
        
        cat_features = X_train.select_dtypes(
            include=['object', 'string']
        ).columns.tolist()
        
        cat_features.append("MSSubClass")
        
        cat_features = list(range(len(cat_features)))
        
        model = get_model(
            'catboost',
            seed=seed,
            **model_params,
        )
        
        model.fit(
            X_train_processed,
            y_train,
            cat_features=cat_features
        )
        
        oof[val_idx] = model.predict(X_val_processed)
        
        test_preds += (
            model.predict(test_processed) / n_splits
        )
        
    return oof, test_preds


def get_oof_dnn(X, y, test, hidden_dim, epochs, batch_size,
                lr, weight_decay, n_splits=5, seed=42, 
                use_total_sf=True,use_total_bathrooms=True, 
                use_house_age=True):
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        torch.manual_seed(seed + fold)
        
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        y_train = y.iloc[train_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type='standard',
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        test_processed = preprocessor.transform(test)
        
        X_train_tensor = torch.tensor(
            np.asarray(X_train_processed), 
            dtype=torch.float32,
        )
        
        y_train_tensor = torch.tensor(
            y_train.to_numpy(),
            dtype=torch.float32,
        )
        
        X_val_tensor = torch.tensor(
            np.asarray(X_val_processed),
            dtype=torch.float32,
        )
        
        X_test_tensor = torch.tensor(
            np.asarray(test_processed),
            dtype=torch.float32,
        )
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        model = HousePricesMLP(
            input_dim=X_train_processed.shape[1],
            hidden_dim=hidden_dim        
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        loss_fn = nn.MSELoss()
        
        for _ in range(epochs):
            model.train()
            
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                
                preds = model(x_batch)
                
                loss = loss_fn(
                    preds, y_batch
                )
                
                loss.backward()
                optimizer.step()
                
        model.eval()
        
        with torch.no_grad():
            
            val_preds = model(X_val_tensor).numpy()
            
            fold_test_preds = model(
                X_test_tensor
            ).numpy()
            
        oof[val_idx] = val_preds
        
        test_preds += (
            fold_test_preds / n_splits
        )
        
    return oof, test_preds


class HousePricesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
