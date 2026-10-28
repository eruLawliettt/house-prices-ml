from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error
from src.models import get_model
from src.preprocessing import build_preprocessor
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

def evaluate_model(estimator, X, y, n_splits=5, 
                   seed=42, n_jobs=1, scoring='neg_root_mean_squared_error'):
    """Evaluate an estimator using cross-validation."""
    
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    
    rmse = -scores
    
    return {
        'fold_rmse': rmse,
        'mean_rmse': rmse.mean(),
        'std_rmse': rmse.std(),
    }
    
def evaluate_catboost(X, y, model_params=None, n_splits=5, seed=42, early_stopping_rounds=None,
                      use_total_sf=False, use_total_bathrooms=False, use_house_age=False):
    
    if model_params is None:
        model_params = {}
        
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    fold_rmse = []
    best_iterations = []
    
    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type="catboost",
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        n_cat_features = (
            len(
                X_train.select_dtypes(
                    include=["object", "string"]
                ).columns
            )
            + 1  # MSSubClass
        )
        
        cat_features = list(range(n_cat_features))
        
        model = get_model(
            'catboost',
            seed=seed,
            **model_params
        )
        
        model.fit(
            X_train_processed,
            y_train,
            cat_features=cat_features,
            eval_set=(X_val_processed, y_val),
            early_stopping_rounds=early_stopping_rounds,
        )
        
        best_iterations.append(model.get_best_iteration())
        
        preds = model.predict(X_val_processed)
        
        rmse = root_mean_squared_error(y_val, preds)
        
        fold_rmse.append(rmse)
        
    fold_rmse = np.array(fold_rmse)
    best_iterations = np.array(best_iterations)
        
    return {
        "fold_rmse": fold_rmse,
        "mean_rmse": fold_rmse.mean(),
        "std_rmse": fold_rmse.std(),
        "mean_best_iteration": best_iterations,
    }

def evaluate_dnn(X, y, model_params=None,
                 n_splits=5, seed=42, epochs=100, batch_size=32, lr=1e-3, weight_decay=0.0,
                 use_total_sf=False, use_total_bathrooms=False, use_house_age=False,):
    
    torch.manual_seed=seed
    np.random.seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if model_params is None:
        model_params = {}
        
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    fold_rmse = []
    
    for train_idx, val_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        preprocessor = build_preprocessor(
            X_train,
            model_type='standard',
            use_total_sf=use_total_sf,
            use_total_bathrooms=use_total_bathrooms,
            use_house_age=use_house_age,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        X_train_tensor = torch.tensor(np.asarray(X_train_processed),  dtype=torch.float32, )
        X_val_tensor = torch.tensor(np.asarray(X_val_processed), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = HousePricesMLP(
            input_dim=X_train_processed.shape[1],
            **model_params,
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                preds = model(x_batch)
                
                loss = criterion(preds, y_batch)
                
                loss.backward()
                optimizer.step()
                
        model.eval()
        
        with torch.no_grad():
            preds = model(X_val_tensor)
            
            val_mse = criterion(preds, y_val_tensor)
            
            val_rmse = torch.sqrt(val_mse).item()
        
        fold_rmse.append(val_rmse)
        
    fold_rmse = np.array(fold_rmse)
    
    return {
        "fold_rmse": fold_rmse,
        "mean_rmse": fold_rmse.mean(),
        "std_rmse": fold_rmse.std(),
    }
    
def train_test_final_dnn(X_train, y_train, X_test, hidden_dim=32, 
                         epochs=100, batch_size=32, lr=3e-3, 
                         weight_decay=0, log_transform=False, seed=42):
    torch.manual_seed(seed)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = HousePricesMLP(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay)
    
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            
            loss.backward()
            optimizer.step()
            
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()
        
    return np.expm1(preds) if log_transform else preds
    
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
