"""
Enhanced Machine Learning Module with Multi-Framework Support
TensorFlow, PyTorch, Prophet, NVIDIA GPU Acceleration + Advanced Algorithms
Complete implementation with all enhancements
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import joblib
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries - Enhanced with additional algorithms
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, silhouette_score, 
    calinski_harabasz_score, explained_variance_score, roc_auc_score
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Advanced ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    logger.info("✅ XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    logger.info("✅ LightGBM available")
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("⚠️ LightGBM not available")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
    logger.info("✅ CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("⚠️ CatBoost not available")

# Deep Learning Frameworks
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("⚠️ TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
    logger.info("✅ PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available")

try:
    from fbprophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("✅ Prophet available")
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠️ Prophet not available")

# NVIDIA GPU Acceleration
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLogistic
    from cuml.cluster import KMeans as cuKMeans
    from cuml import PCA as cuPCA
    CUML_AVAILABLE = True
    logger.info("✅ NVIDIA cuML available for GPU acceleration")
except ImportError:
    CUML_AVAILABLE = False

# Additional advanced libraries
try:
    import umap
    UMAP_AVAILABLE = True
    logger.info("✅ UMAP available for dimensionality reduction")
except ImportError:
    UMAP_AVAILABLE = False

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
    logger.info("✅ HDBSCAN available for advanced clustering")
except ImportError:
    HDBSCAN_AVAILABLE = False

class PyTorchMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron for classification and regression"""
    def __init__(self, input_size, output_size=1, hidden_sizes=[128, 64, 32], task_type='classification'):
        super(PyTorchMLP, self).__init__()
        self.task_type = task_type
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        if self.task_type == 'classification' and output.shape[1] == 1:
            return torch.sigmoid(output)
        elif self.task_type == 'classification' and output.shape[1] > 1:
            return torch.softmax(output, dim=1)
        return output

class MLModule:
    """Enhanced ML Module with Multi-Framework Support and AutoML Pipeline"""
    
    def __init__(self, config=None):
        self.config = config
        self.use_gpu = self._detect_gpu_capability()
        self.device = self._setup_device()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = ""
        self.task_type = ""
        self.clustering_results = {}
        self.dimensionality_reduction = {}
        
        # Create models directory
        Path("models/trained_models").mkdir(parents=True, exist_ok=True)
        Path("models/artifacts").mkdir(parents=True, exist_ok=True)
        Path("models/clustering").mkdir(parents=True, exist_ok=True)
        Path("models/dim_reduction").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced ML Module initialized - GPU: {self.use_gpu}, Device: {self.device}")
    
    def _detect_gpu_capability(self) -> bool:
        """Enhanced GPU detection for multiple frameworks"""
        gpu_available = False
        
        # Check TensorFlow GPU
        if TENSORFLOW_AVAILABLE:
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_available:
                logger.info("✅ TensorFlow GPU detected")
        
        # Check PyTorch GPU
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            logger.info(f"✅ PyTorch GPU detected: {torch.cuda.get_device_name()}")
        
        # Check cuML
        if CUML_AVAILABLE:
            gpu_available = True
            logger.info("✅ NVIDIA cuML GPU detected")
        
        return gpu_available
    
    def _setup_device(self):
        """Setup appropriate device for PyTorch"""
        if PYTORCH_AVAILABLE:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return None

    def auto_ml_pipeline(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        task_type: str = 'auto',
                        test_size: float = 0.2,
                        cv_folds: int = 5,
                        random_state: int = 42,
                        include_advanced_models: bool = True) -> Dict[str, Any]:
        """
        Complete automated ML pipeline
        
        Args:
            X: Features dataframe
            y: Target variable
            task_type: 'classification', 'regression', or 'auto'
            test_size: Test set size
            cv_folds: Cross-validation folds
            random_state: Random state for reproducibility
            include_advanced_models: Whether to include advanced feature engineering
            
        Returns:
            Dict containing all ML results
        """
        try:
            logger.info("Starting Enhanced Auto-ML Pipeline")
            
            # Detect task type
            if task_type == 'auto':
                self.task_type = self._detect_task_type(y)
            else:
                self.task_type = task_type
            
            logger.info(f"Task Type: {self.task_type}")
            
            # Store feature and target names
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, 'name') else 'target'
            
            # Store y for model creation
            self.y_encoded = y
            
            # Enhanced preprocessing with feature engineering
            X_processed, y_processed = self._enhanced_preprocess_for_ml(X, y, include_advanced_models)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=random_state,
                stratify=y_processed if self.task_type == 'classification' and len(np.unique(y_processed)) > 1 else None
            )
            
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Train multiple models including advanced frameworks
            model_results = self._train_enhanced_models(X_train, y_train, X_test, y_test, cv_folds)
            
            # Select best model
            best_model_name = self._select_best_model(model_results)
            
            # Generate feature importance
            feature_importance = self._get_feature_importance(best_model_name, X_train.columns)
            
            # Generate predictions
            best_model = self.models[best_model_name]
            train_predictions, test_predictions = self._get_predictions(best_model, X_train, X_test)
            
            # Calculate final metrics
            train_metrics = self._calculate_metrics(y_train, train_predictions)
            test_metrics = self._calculate_metrics(y_test, test_predictions)
            
            # Create results summary
            results = {
                'task_type': self.task_type,
                'best_model': best_model_name,
                'best_model_object': best_model,
                'all_models': model_results,
                'feature_importance': feature_importance,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'data_info': {
                    'n_features': len(self.feature_names),
                    'n_samples_train': len(X_train),
                    'n_samples_test': len(X_test),
                    'feature_names': self.feature_names,
                    'target_name': self.target_name
                },
                'predictions': {
                    'train_predictions': train_predictions,
                    'test_predictions': test_predictions,
                    'train_actual': y_train,
                    'test_actual': y_test
                }
            }
            
            # Save best model
            self._save_model(best_model_name, results)
            
            logger.info(f"✅ Enhanced Auto-ML Pipeline completed - Best model: {best_model_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in Enhanced Auto-ML Pipeline: {str(e)}")
            raise e

    def _enhanced_preprocess_for_ml(self, X: pd.DataFrame, y: pd.Series, include_advanced_models: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced preprocessing with feature engineering and clustering features"""
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values in features
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if X_processed[col].dtype in ['int64', 'float64']:
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Encode categorical features
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_processed[col] = self.encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
        
        # Handle missing values in target
        if y_processed.isnull().any():
            if self.task_type == 'classification':
                y_processed.fillna(y_processed.mode()[0] if len(y_processed.mode()) > 0 else 0, inplace=True)
            else:
                y_processed.fillna(y_processed.median(), inplace=True)
        
        # Encode target if classification
        if self.task_type == 'classification' and y_processed.dtype == 'object':
            if 'target_encoder' not in self.encoders:
                self.encoders['target_encoder'] = LabelEncoder()
                y_processed = pd.Series(self.encoders['target_encoder'].fit_transform(y_processed))
        
        # Advanced feature engineering
        if include_advanced_models:
            X_processed = self._add_advanced_features(X_processed, y_processed)
        
        # Scale features for algorithms that need it
        if 'feature_scaler' not in self.scalers:
            self.scalers['feature_scaler'] = StandardScaler()
            X_scaled = self.scalers['feature_scaler'].fit_transform(X_processed)
            X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        return X_processed, y_processed

    def _add_advanced_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Add advanced features like clustering results and dimensionality reduction"""
        X_enhanced = X.copy()
        
        try:
            # Add K-Means clustering features
            if len(X) > 10:  # Only if we have enough samples
                kmeans_features = self._add_clustering_features(X)
                X_enhanced = pd.concat([X_enhanced, kmeans_features], axis=1)
            
            # Add PCA components
            pca_features = self._add_pca_features(X)
            X_enhanced = pd.concat([X_enhanced, pca_features], axis=1)
            
            # Add statistical features
            statistical_features = self._add_statistical_features(X)
            X_enhanced = pd.concat([X_enhanced, statistical_features], axis=1)
            
            logger.info(f"✅ Added {len(X_enhanced.columns) - len(X.columns)} advanced features")
            
        except Exception as e:
            logger.warning(f"⚠️ Advanced feature engineering failed: {str(e)}")
        
        return X_enhanced

    def _add_clustering_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add clustering-based features"""
        clustering_features = pd.DataFrame(index=X.index)
        
        try:
            # K-Means clustering with multiple k values
            for k in [2, 3, 4]:
                if CUML_AVAILABLE and self.use_gpu:
                    kmeans = cuKMeans(n_clusters=k, random_state=42)
                    clusters = kmeans.fit_predict(X)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X)
                
                clustering_features[f'kmeans_cluster_{k}'] = clusters
                clustering_features[f'kmeans_distance_{k}'] = kmeans.transform(X).min(axis=1)
            
            # Store clustering results for explainability
            self.clustering_results['kmeans'] = {
                'model': kmeans,
                'n_clusters': k,
                'inertia': kmeans.inertia_ if hasattr(kmeans, 'inertia_') else None
            }
            
        except Exception as e:
            logger.warning(f"⚠️ K-Means clustering failed: {str(e)}")
        
        return clustering_features

    def _add_pca_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add PCA components as features"""
        pca_features = pd.DataFrame(index=X.index)
        
        try:
            n_components = min(5, X.shape[1])
            
            if CUML_AVAILABLE and self.use_gpu:
                pca = cuPCA(n_components=n_components)
                components = pca.fit_transform(X)
            else:
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X)
            
            for i in range(n_components):
                pca_features[f'pca_component_{i+1}'] = components[:, i]
            
            # Store PCA results
            self.dimensionality_reduction['pca'] = {
                'model': pca,
                'explained_variance_ratio': pca.explained_variance_ratio_ if hasattr(pca, 'explained_variance_ratio_') else None,
                'n_components': n_components
            }
            
        except Exception as e:
            logger.warning(f"⚠️ PCA failed: {str(e)}")
        
        return pca_features

    def _add_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        statistical_features = pd.DataFrame(index=X.index)
        
        try:
            # Row-wise statistics
            statistical_features['row_mean'] = X.mean(axis=1)
            statistical_features['row_std'] = X.std(axis=1)
            statistical_features['row_skew'] = X.skew(axis=1)
            statistical_features['row_kurtosis'] = X.kurtosis(axis=1)
            
            # Interaction features (example)
            if len(X.columns) >= 2:
                statistical_features['feature_interaction_1'] = X.iloc[:, 0] * X.iloc[:, 1]
            
        except Exception as e:
            logger.warning(f"⚠️ Statistical features failed: {str(e)}")
        
        return statistical_features

    def _get_enhanced_model_algorithms(self) -> Dict[str, Any]:
        """Get enhanced models including TensorFlow and PyTorch architectures"""
        algorithms = self._get_extended_model_algorithms()  # Extended with new algorithms
        
        # Add TensorFlow models if available
        if TENSORFLOW_AVAILABLE:
            if self.task_type == 'classification':
                algorithms['TensorFlow_DNN'] = self._create_tf_classifier()
            else:
                algorithms['TensorFlow_Regressor'] = self._create_tf_regressor()
        
        # Add PyTorch models if available
        if PYTORCH_AVAILABLE:
            output_size = 1 if self.task_type == 'regression' else len(np.unique(self.y_encoded))
            algorithms['PyTorch_MLP'] = PyTorchMLP(
                input_size=len(self.feature_names), 
                output_size=output_size,
                task_type=self.task_type
            )
        
        # Add GPU-accelerated models
        if CUML_AVAILABLE:
            if self.task_type == 'classification':
                algorithms['GPU_RandomForest'] = cuRF()
                algorithms['GPU_Logistic'] = cuLogistic()
        
        return algorithms

    def _get_extended_model_algorithms(self) -> Dict[str, Any]:
        """Get extended ML algorithms including KNN, Decision Trees, etc."""
        algorithms = {}
        
        if self.task_type == 'classification':
            # Basic algorithms
            algorithms['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            algorithms['LogisticRegression'] = LogisticRegression(random_state=42, max_iter=1000)
            algorithms['SVM'] = SVC(random_state=42, probability=True)
            
            # NEW: Decision Tree
            algorithms['DecisionTree'] = DecisionTreeClassifier(random_state=42, max_depth=10)
            
            # NEW: K-Nearest Neighbors
            algorithms['KNN'] = KNeighborsClassifier(n_neighbors=5)
            
            # NEW: Gaussian Naive Bayes
            algorithms['GaussianNB'] = GaussianNB()
            
            # NEW: Gradient Boosting
            algorithms['GradientBoosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            # NEW: AdaBoost
            algorithms['AdaBoost'] = AdaBoostClassifier(n_estimators=100, random_state=42)
            
            # Advanced algorithms (if available)
            if XGB_AVAILABLE:
                algorithms['XGBoost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0
                )
            
            if LGB_AVAILABLE:
                algorithms['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
            
            if CATBOOST_AVAILABLE:
                algorithms['CatBoost'] = CatBoostClassifier(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
        
        else:  # regression
            # Basic algorithms
            algorithms['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            algorithms['LinearRegression'] = LinearRegression()
            algorithms['Ridge'] = Ridge(random_state=42)
            algorithms['Lasso'] = Lasso(random_state=42)
            algorithms['SVR'] = SVR()
            
            # NEW: Decision Tree Regressor
            algorithms['DecisionTree'] = DecisionTreeRegressor(random_state=42, max_depth=10)
            
            # NEW: K-Nearest Neighbors Regressor
            algorithms['KNN'] = KNeighborsRegressor(n_neighbors=5)
            
            # NEW: Gradient Boosting Regressor
            algorithms['GradientBoosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # NEW: AdaBoost Regressor
            algorithms['AdaBoost'] = AdaBoostRegressor(n_estimators=100, random_state=42)
            
            # NEW: ElasticNet
            algorithms['ElasticNet'] = ElasticNet(random_state=42)
            
            # Advanced algorithms (if available)
            if XGB_AVAILABLE:
                algorithms['XGBoost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    verbosity=0
                )
            
            if LGB_AVAILABLE:
                algorithms['LightGBM'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
            
            if CATBOOST_AVAILABLE:
                algorithms['CatBoost'] = CatBoostRegressor(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
        
        return algorithms

    def _train_enhanced_models(self, X_train, y_train, X_test, y_test, cv_folds) -> Dict[str, Any]:
        """Train multiple ML algorithms including advanced frameworks and return results"""
        
        algorithms = self._get_enhanced_model_algorithms()
        results = {}
        
        for name, model in algorithms.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model based on type
                model_results = self._train_advanced_models(X_train, y_train, X_test, y_test, name, model)
                
                if model_results:
                    results[name] = model_results
                    self.models[name] = model
                    logger.info(f"✅ {name} completed - Score: {results[name]['primary_score']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {str(e)}")
                continue
        
        return results

    def _train_advanced_models(self, X_train, y_train, X_test, y_test, model_name, model):
        """Enhanced training for different framework models"""
        start_time = time.time()
        
        try:
            if TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                # TensorFlow training
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,  # Reduced for faster training
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    ]
                )
                
                # Get predictions
                train_pred = model.predict(X_train).flatten()
                test_pred = model.predict(X_test).flatten()
                
                if self.task_type == 'classification':
                    train_pred = (train_pred > 0.5).astype(int)
                    test_pred = (test_pred > 0.5).astype(int)
                
                training_time = time.time() - start_time
                
            elif PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                # PyTorch training
                train_pred, test_pred, training_time = self._train_pytorch_model(model, X_train, y_train, X_test, y_test)
                
            else:
                # Traditional scikit-learn style models
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            # Cross-validation for traditional models
            cv_scores = np.array([0])  # Default
            cv_mean = 0
            cv_std = 0
            
            if not (TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model)) and \
               not (PYTORCH_AVAILABLE and isinstance(model, nn.Module)):
                try:
                    scorer = 'accuracy' if self.task_type == 'classification' else 'r2'
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scorer)  # Reduced folds for speed
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    pass
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred)
            test_metrics = self._calculate_metrics(y_test, test_pred)
            
            # Get primary score
            primary_score = test_metrics.get('accuracy', test_metrics.get('r2', 0))
            
            return {
                'model': model,
                'training_time': training_time,
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'primary_score': primary_score,
                'gpu_accelerated': self._is_gpu_accelerated(model)
            }
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {str(e)}")
            return None

    def _train_pytorch_model(self, model, X_train, y_train, X_test, y_test):
        """Train PyTorch model with GPU support"""
        start_time = time.time()
        model.to(self.device)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
        
        if self.device.type == 'cuda':
            X_train_tensor = X_train_tensor.cuda()
            y_train_tensor = y_train_tensor.cuda()
            X_test_tensor = X_test_tensor.cuda()
        
        # Training setup
        if self.task_type == 'classification':
            criterion = nn.BCELoss() if model.network[-1].out_features == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):  # Reduced epochs for faster training
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            
            if self.task_type == 'classification' and outputs.shape[1] == 1:
                loss = criterion(outputs.squeeze(), y_train_tensor)
            else:
                loss = criterion(outputs, y_train_tensor.unsqueeze(1) if self.task_type == 'regression' else y_train_tensor.long())
                
            loss.backward()
            optimizer.step()
        
        # Predictions
        model.eval()
        with torch.no_grad():
            train_output = model(X_train_tensor)
            test_output = model(X_test_tensor)
            
            if self.task_type == 'classification':
                if train_output.shape[1] == 1:
                    train_pred = (train_output.squeeze().cpu().numpy() > 0.5).astype(int)
                    test_pred = (test_output.squeeze().cpu().numpy() > 0.5).astype(int)
                else:
                    train_pred = torch.argmax(train_output, dim=1).cpu().numpy()
                    test_pred = torch.argmax(test_output, dim=1).cpu().numpy()
            else:
                train_pred = train_output.squeeze().cpu().numpy()
                test_pred = test_output.squeeze().cpu().numpy()
        
        training_time = time.time() - start_time
        return train_pred, test_pred, training_time

    def _get_predictions(self, model, X_train, X_test):
        """Handle predictions for different model types"""
        if TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
            train_predictions = model.predict(X_train).flatten()
            test_predictions = model.predict(X_test).flatten()
            if self.task_type == 'classification':
                train_predictions = (train_predictions > 0.5).astype(int)
                test_predictions = (test_predictions > 0.5).astype(int)
        elif PYTORCH_AVAILABLE and isinstance(model, nn.Module):
            train_predictions = self._pytorch_predict(model, X_train)
            test_predictions = self._pytorch_predict(model, X_test)
        else:
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
        
        return train_predictions, test_predictions

    def _pytorch_predict(self, model, X):
        """Make predictions with PyTorch model"""
        model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        
        if self.device.type == 'cuda':
            X_tensor = X_tensor.cuda()
            
        with torch.no_grad():
            outputs = model(X_tensor)
            
            if self.task_type == 'classification':
                if outputs.shape[1] == 1:
                    predictions = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
                else:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
                
        return predictions

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate metrics based on task type"""
        
        if self.task_type == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:
            return {
                'r2': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }

    def _select_best_model(self, results: Dict[str, Any]) -> str:
        """Select the best performing model"""
        
        if not results:
            return None
        
        # Sort by primary score (accuracy for classification, r2 for regression)
        best_model = max(results.keys(), key=lambda x: results[x]['primary_score'])
        
        return best_model

    def _get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the model"""
        
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
            
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) == 1:
                    importances = np.abs(model.coef_)
                else:
                    importances = np.abs(model.coef_).mean(axis=0)
                importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {str(e)}")
        
        return importance_dict

    def _is_gpu_accelerated(self, model) -> bool:
        """Check if model is GPU accelerated"""
        
        model_name = type(model).__name__
        
        # Check for GPU-enabled versions
        gpu_indicators = ['XGB', 'LightGBM', 'CatBoost', 'cuML']
        
        for indicator in gpu_indicators:
            if indicator in model_name:
                return True
        
        # Check if it's a PyTorch model on GPU
        if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
            return next(model.parameters()).is_cuda
        
        # Check if it's a TensorFlow model on GPU
        if TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
            return any('GPU' in device for device in [d.device_type for d in model._distribution_strategy.extended.parameter_devices])
        
        return False

    def _save_model(self, model_name: str, results: Dict[str, Any]):
        """Save trained model and artifacts"""
        
        try:
            # Save model
            model_path = f"models/trained_models/{model_name}_model.joblib"
            
            # Handle different model types
            model_to_save = self.models[model_name]
            
            if TENSORFLOW_AVAILABLE and isinstance(model_to_save, tf.keras.Model):
                model_to_save.save(f"models/trained_models/{model_name}_model.h5")
            elif PYTORCH_AVAILABLE and isinstance(model_to_save, nn.Module):
                torch.save(model_to_save.state_dict(), f"models/trained_models/{model_name}_model.pth")
            else:
                joblib.dump(model_to_save, model_path)
            
            # Save preprocessing artifacts
            artifacts = {
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'task_type': self.task_type
            }
            
            artifacts_path = f"models/artifacts/{model_name}_artifacts.joblib"
            joblib.dump(artifacts, artifacts_path)
            
            # Save results summary
            results_summary = {
                'model_name': model_name,
                'task_type': self.task_type,
                'best_score': results['test_metrics'],
                'feature_importance': results['feature_importance'],
                'training_timestamp': pd.Timestamp.now().isoformat()
            }
            
            results_path = f"models/artifacts/{model_name}_results.joblib"
            joblib.dump(results_summary, results_path)
            
            logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")

    def _detect_task_type(self, y: pd.Series) -> str:
        """Automatically detect if it's classification or regression"""
        
        if y.dtype == 'object' or y.dtype == 'category':
            return 'classification'
        elif y.nunique() <= 20 and y.dtype in ['int64', 'int32']:
            return 'classification'
        else:
            return 'regression'

    def _preprocess_for_ml(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Original preprocessing method for backward compatibility"""
        return self._enhanced_preprocess_for_ml(X, y, include_advanced_models=False)

    def _get_model_algorithms(self) -> Dict[str, Any]:
        """Original model algorithms for backward compatibility"""
        return self._get_extended_model_algorithms()

    def _create_tf_classifier(self):
        """Create TensorFlow DNN classifier"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _create_tf_regressor(self):
        """Create TensorFlow DNN regressor"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    # NEW: CLUSTERING METHODS
    def perform_clustering_analysis(self, 
                                  X: pd.DataFrame, 
                                  n_clusters: int = 3,
                                  method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering analysis using various algorithms
        
        Args:
            X: Features dataframe
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Dictionary with clustering results
        """
        try:
            logger.info(f"Performing {method} clustering analysis...")
            
            # Preprocess data
            X_processed, _ = self._preprocess_for_ml(X, pd.Series([0]*len(X)))  # Dummy target
            
            if method == 'kmeans':
                if CUML_AVAILABLE and self.use_gpu:
                    model = cuKMeans(n_clusters=n_clusters, random_state=42)
                else:
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
            elif method == 'dbscan' and len(X) < 10000:  # DBSCAN can be slow on large datasets
                model = DBSCAN(eps=0.5, min_samples=5)
                
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                
            else:
                return {'error': f'Unsupported clustering method: {method}'}
            
            # Fit and predict
            clusters = model.fit_predict(X_processed)
            
            # Calculate metrics
            metrics = {}
            if len(np.unique(clusters)) > 1:
                try:
                    metrics['silhouette_score'] = silhouette_score(X_processed, clusters)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_processed, clusters)
                except:
                    pass
            
            results = {
                'model': model,
                'clusters': clusters,
                'n_clusters_found': len(np.unique(clusters)),
                'metrics': metrics,
                'cluster_sizes': pd.Series(clusters).value_counts().to_dict()
            }
            
            # Store results
            self.clustering_results[method] = results
            
            logger.info(f"✅ {method} clustering completed - Found {len(np.unique(clusters))} clusters")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in clustering analysis: {str(e)}")
            return {'error': str(e)}

    # NEW: DIMENSIONALITY REDUCTION METHODS
    def perform_dimensionality_reduction(self, 
                                       X: pd.DataFrame, 
                                       method: str = 'pca',
                                       n_components: int = 2) -> Dict[str, Any]:
        """
        Perform dimensionality reduction
        
        Args:
            X: Features dataframe
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to keep
            
        Returns:
            Dictionary with reduction results
        """
        try:
            logger.info(f"Performing {method} dimensionality reduction...")
            
            # Preprocess data
            X_processed, _ = self._preprocess_for_ml(X, pd.Series([0]*len(X)))
            
            if method == 'pca':
                if CUML_AVAILABLE and self.use_gpu:
                    model = cuPCA(n_components=n_components)
                else:
                    model = PCA(n_components=n_components)
                    
            elif method == 'tsne':
                model = TSNE(n_components=n_components, random_state=42)
                
            elif method == 'umap' and UMAP_AVAILABLE:
                model = umap.UMAP(n_components=n_components, random_state=42)
                
            else:
                return {'error': f'Unsupported dimensionality reduction method: {method}'}
            
            # Transform data
            components = model.fit_transform(X_processed)
            
            results = {
                'model': model,
                'components': components,
                'explained_variance': model.explained_variance_ratio_ if hasattr(model, 'explained_variance_ratio_') else None,
                'method': method,
                'n_components': n_components
            }
            
            # Store results
            self.dimensionality_reduction[method] = results
            
            logger.info(f"✅ {method} dimensionality reduction completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in dimensionality reduction: {str(e)}")
            return {'error': str(e)}

    # NEW: ENHANCED EXPLAINABILITY METHODS
    def get_model_explainability(self, 
                               model_name: str, 
                               X: pd.DataFrame,
                               y: pd.Series = None) -> Dict[str, Any]:
        """
        Get comprehensive model explainability
        
        Args:
            model_name: Name of the trained model
            X: Features dataframe
            y: Target variable (optional)
            
        Returns:
            Dictionary with explainability results
        """
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        explainability = {}
        
        try:
            # Feature importance
            explainability['feature_importance'] = self._get_feature_importance(model_name, X.columns)
            
            # Model complexity
            explainability['model_complexity'] = self._get_model_complexity(model)
            
            logger.info(f"✅ Explainability analysis completed for {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error in explainability analysis: {str(e)}")
            explainability['error'] = str(e)
        
        return explainability

    def _get_model_complexity(self, model) -> Dict[str, Any]:
        """Get model complexity metrics"""
        complexity = {}
        
        try:
            model_type = type(model).__name__
            
            if hasattr(model, 'n_estimators'):
                complexity['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                complexity['max_depth'] = model.max_depth
            if hasattr(model, 'n_features_in_'):
                complexity['n_features'] = model.n_features_in_
            if hasattr(model, 'coef_'):
                complexity['n_coefficients'] = len(model.coef_.flatten())
            
            complexity['model_type'] = model_type
            
        except Exception as e:
            logger.warning(f"⚠️ Model complexity analysis failed: {str(e)}")
        
        return complexity

    # NEW: FEATURE SELECTION METHODS
    def perform_feature_selection(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                method: str = 'mutual_info',
                                k: int = 10) -> Dict[str, Any]:
        """
        Perform feature selection
        
        Args:
            X: Features dataframe
            y: Target variable
            method: Selection method ('mutual_info', 'f_classif', 'variance')
            k: Number of top features to select
            
        Returns:
            Dictionary with feature selection results
        """
        try:
            logger.info(f"Performing {method} feature selection...")
            
            X_processed, y_processed = self._preprocess_for_ml(X, y)
            
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_processed.shape[1]))
            elif method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=min(k, X_processed.shape[1]))
            else:
                return {'error': f'Unsupported feature selection method: {method}'}
            
            # Fit selector
            X_selected = selector.fit_transform(X_processed, y_processed)
            
            # Get selected features and scores
            feature_scores = dict(zip(X_processed.columns, selector.scores_))
            selected_features = X_processed.columns[selector.get_support()].tolist()
            
            results = {
                'selector': selector,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'X_selected': X_selected,
                'method': method,
                'k': k
            }
            
            logger.info(f"✅ Feature selection completed - Selected {len(selected_features)} features")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in feature selection: {str(e)}")
            return {'error': str(e)}

    # PROPHET FORECASTING METHOD
    def create_prophet_forecast(self, 
                              df: pd.DataFrame, 
                              date_column: str,
                              target_column: str,
                              periods: int = 30,
                              frequency: str = 'D') -> Dict[str, Any]:
        """
        Create time series forecasts using Facebook Prophet
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the date column
            target_column: Name of the target column
            periods: Number of periods to forecast
            frequency: Frequency of the time series ('D', 'M', 'Y', etc.)
            
        Returns:
            Dictionary with forecast results
        """
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet not available. Install with: pip install fbprophet'}
        
        try:
            # Prepare data for Prophet
            prophet_df = df[[date_column, target_column]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Add additional seasonalities if enough data
            if len(prophet_df) > 730:  # More than 2 years
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            logger.info("Fitting Prophet model...")
            model.fit(prophet_df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=periods, freq=frequency)
            
            # Forecast
            forecast = model.predict(future)
            
            # Calculate performance metrics on historical data
            historical_forecast = forecast[forecast['ds'].isin(prophet_df['ds'])]
            y_true = prophet_df['y'].values
            y_pred = historical_forecast['yhat'].values
            
            metrics = {
                'mae': np.mean(np.abs(y_true - y_pred)),
                'mse': np.mean((y_true - y_pred) ** 2),
                'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
            }
            
            results = {
                'model': model,
                'forecast': forecast,
                'historical_data': prophet_df,
                'metrics': metrics
            }
            
            logger.info("✅ Prophet forecasting completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in Prophet forecasting: {str(e)}")
            return {'error': str(e)}

    # HYPERPARAMETER TUNING METHOD
    def hyperparameter_tuning(self, 
                             model_name: str, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name: Name of the model to tune
            X: Features
            y: Target
            param_grid: Parameters to tune
            
        Returns:
            Dict with tuning results
        """
        try:
            algorithms = self._get_enhanced_model_algorithms()
            
            if model_name not in algorithms:
                return {'error': f'Model {model_name} not available'}
            
            model = algorithms[model_name]
            
            # Skip tuning for deep learning models
            if (TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model)) or \
               (PYTORCH_AVAILABLE and isinstance(model, nn.Module)):
                return {'error': 'Hyperparameter tuning not supported for deep learning models in this version'}
            
            # Preprocess data
            X_processed, y_processed = self._preprocess_for_ml(X, y)
            
            # Grid search
            scorer = 'accuracy' if self.task_type == 'classification' else 'r2'
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            grid_search.fit(X_processed, y_processed)
            
            results = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            # Update stored model
            self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
            
            logger.info(f"✅ Hyperparameter tuning completed - Best score: {grid_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in hyperparameter tuning: {str(e)}")
            return {'error': str(e)}

    # PREDICTION METHOD
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Preprocess features
            X_processed = X.copy()
            
            # Handle missing values
            for col in X_processed.columns:
                if X_processed[col].isnull().any():
                    if X_processed[col].dtype in ['int64', 'float64']:
                        X_processed[col].fillna(X_processed[col].median(), inplace=True)
                    else:
                        X_processed[col].fillna('Unknown', inplace=True)
            
            # Encode categorical features
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in self.encoders:
                    try:
                        X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        X_processed[col] = 0
            
            # Scale features
            if 'feature_scaler' in self.scalers:
                X_scaled = self.scalers['feature_scaler'].transform(X_processed)
                X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
            
            # Make predictions
            model = self.models[model_name]
            
            if TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                predictions = model.predict(X_processed).flatten()
                if self.task_type == 'classification':
                    predictions = (predictions > 0.5).astype(int)
            elif PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                predictions = self._pytorch_predict(model, X_processed)
            else:
                predictions = model.predict(X_processed)
            
            # Decode predictions if classification
            if self.task_type == 'classification' and 'target_encoder' in self.encoders:
                predictions = self.encoders['target_encoder'].inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error making predictions: {str(e)}")
            return np.array([])

    # MODEL LOADING METHOD
    def load_model(self, model_name: str) -> bool:
        """Load a previously trained model"""
        
        try:
            # Try different model formats
            model_paths = [
                f"models/trained_models/{model_name}_model.joblib",
                f"models/trained_models/{model_name}_model.h5",
                f"models/trained_models/{model_name}_model.pth"
            ]
            
            model_loaded = False
            
            for model_path in model_paths:
                if Path(model_path).exists():
                    if model_path.endswith('.h5') and TENSORFLOW_AVAILABLE:
                        self.models[model_name] = tf.keras.models.load_model(model_path)
                    elif model_path.endswith('.pth') and PYTORCH_AVAILABLE:
                        # For PyTorch, we need to recreate the model architecture first
                        # This is a simplified version - you might need to adjust based on your model
                        input_size = len(self.feature_names) if self.feature_names else 10
                        output_size = 1 if self.task_type == 'regression' else 2
                        model = PyTorchMLP(input_size=input_size, output_size=output_size, task_type=self.task_type)
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        self.models[model_name] = model
                    else:
                        self.models[model_name] = joblib.load(model_path)
                    
                    model_loaded = True
                    break
            
            if not model_loaded:
                logger.error(f"❌ Model file not found for: {model_name}")
                return False
            
            # Load artifacts
            artifacts_path = f"models/artifacts/{model_name}_artifacts.joblib"
            if Path(artifacts_path).exists():
                artifacts = joblib.load(artifacts_path)
                self.scalers = artifacts.get('scalers', {})
                self.encoders = artifacts.get('encoders', {})
                self.feature_names = artifacts.get('feature_names', [])
                self.target_name = artifacts.get('target_name', '')
                self.task_type = artifacts.get('task_type', '')
            
            logger.info(f"✅ Model loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    # MODEL INFO METHOD
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        
        info = {
            'available_models': list(self.models.keys()),
            'task_type': self.task_type,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'gpu_available': self.use_gpu,
            'clustering_results_available': list(self.clustering_results.keys()),
            'dimensionality_reduction_available': list(self.dimensionality_reduction.keys())
        }
        
        return info