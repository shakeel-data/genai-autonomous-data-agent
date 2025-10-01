"""
Enhanced Model Explainability Module
SHAP, LIME, and advanced explanations with business-friendly insights
Complete implementation with all enhancements
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')
# Add these imports if not present
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("✅ SHAP available for model explanations")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("⚠️ SHAP not available")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    logger.info("✅ LIME available for model explanations")
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("⚠️ LIME not available")

# Additional explainability libraries
try:
    from sklearn.inspection import partial_dependence
    PDP_AVAILABLE = True
    logger.info("✅ Partial Dependence Plots available")
except ImportError:
    PDP_AVAILABLE = False

try:
    import dalex as dx
    DALEX_AVAILABLE = True
    logger.info("✅ DALEX available for model explanations")
except ImportError:
    DALEX_AVAILABLE = False

class ExplainModule:
    """Advanced model explainability with SHAP, LIME, and comprehensive insights"""
    
    def __init__(self, config=None):
        self.config = config
        self.explainers = {}
        self.explanations = {}
        
        # Enhanced explainability techniques
        self.available_techniques = {
            'shap': SHAP_AVAILABLE,
            'lime': LIME_AVAILABLE,
            'partial_dependence': PDP_AVAILABLE,
            'dalex': DALEX_AVAILABLE,
            'feature_importance': True,
            'business_insights': True
        }
        
        logger.info("Enhanced Explainability Module initialized")

    def _align_features_with_model(self, model, X: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FIX: Enhanced feature alignment for CatBoost models
        Solves: "At position 1 should be feature with name Order Date (found Ship Mode)"
        """
        try:
            logger.info("🔄 Starting feature alignment for model compatibility...")
            
            # Method 1: Try to get feature names from CatBoost model
            model_feature_names = None
            
            # CatBoost specific feature extraction
            if hasattr(model, 'feature_names_'):
                model_feature_names = model.feature_names_
                logger.info(f"✅ Found {len(model_feature_names)} features in model.feature_names_")
            
            # Method 2: Try get_feature_names method
            elif hasattr(model, 'get_feature_names'):
                try:
                    model_feature_names = model.get_feature_names()
                    logger.info(f"✅ Found {len(model_feature_names)} features via get_feature_names()")
                except Exception as e:
                    logger.warning(f"get_feature_names() failed: {e}")
            
            # Method 3: Try feature_names attribute
            elif hasattr(model, 'feature_names'):
                model_feature_names = model.feature_names
                logger.info(f"✅ Found {len(model_feature_names)} features in model.feature_names")
            
            # Method 4: For CatBoost models, try to extract from internal structure
            elif 'catboost' in str(type(model)).lower():
                try:
                    # CatBoost stores features in _features attribute
                    if hasattr(model, '_features'):
                        model_feature_names = [f.name for f in model._features]
                        logger.info(f"✅ Found {len(model_feature_names)} features in CatBoost _features")
                    # Try to get from feature_importances_ index
                    elif hasattr(model, 'feature_importances_'):
                        model_feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
                        logger.warning(f"⚠️ Using generated feature names: {len(model_feature_names)} features")
                except Exception as e:
                    logger.warning(f"CatBoost feature extraction failed: {e}")
            
            if model_feature_names is None:
                logger.warning("❌ Cannot determine model's feature names, using input features")
                return X
            
            logger.info(f"📊 Model expects features: {model_feature_names[:3]}... (total: {len(model_feature_names)})")
            logger.info(f"📊 Input data has features: {X.columns[:3].tolist()}... (total: {len(X.columns)})")
            
            # CRITICAL: Check for exact feature match
            input_features_set = set(X.columns)
            model_features_set = set(model_feature_names)
            
            missing_in_input = model_features_set - input_features_set
            extra_in_input = input_features_set - model_features_set
            
            if missing_in_input:
                logger.error(f"❌ MISSING FEATURES in input: {list(missing_in_input)}")
                raise ValueError(f"Model requires {len(missing_in_input)} missing features: {list(missing_in_input)}")
            
            if extra_in_input:
                logger.warning(f"⚠️ EXTRA FEATURES in input: {list(extra_in_input)}")
            
            # CRITICAL: Reorder features to EXACT match model's expected order
            aligned_features = []
            missing_features = []
            
            for i, model_feat in enumerate(model_feature_names):
                if model_feat in X.columns:
                    aligned_features.append(model_feat)
                else:
                    missing_features.append(model_feat)
                    logger.error(f"❌ Feature '{model_feat}' at position {i} not found in input data")
            
            if missing_features:
                raise ValueError(f"Missing {len(missing_features)} features required by model: {missing_features}")
            
            # Create aligned dataframe with EXACT feature order
            X_aligned = X[aligned_features].copy()
            
            # Validate alignment
            aligned_features_list = X_aligned.columns.tolist()
            if aligned_features_list == model_feature_names:
                logger.info("✅ SUCCESS: Features perfectly aligned with model expectations!")
            else:
                logger.warning("⚠️ Feature order may not be perfectly aligned")
                logger.info(f"Model order: {model_feature_names[:3]}...")
                logger.info(f"Aligned order: {aligned_features_list[:3]}...")
            
            logger.info(f"🎯 Final aligned features: {len(X_aligned.columns)} features")
            return X_aligned
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Feature alignment failed: {str(e)}")
            logger.info("🔄 Attempting fallback strategy...")
            
            # Fallback: Try to match by name regardless of order
            try:
                if hasattr(model, 'feature_names_'):
                    model_features = model.feature_names_
                    available_features = [f for f in model_features if f in X.columns]
                    
                    if len(available_features) == len(model_features):
                        X_fallback = X[model_features].copy()
                        logger.info("✅ Fallback successful: Features matched by name")
                        return X_fallback
                    else:
                        missing = set(model_features) - set(available_features)
                        logger.error(f"❌ Fallback failed: Missing features {list(missing)}")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
            
            # Last resort: return original with warning
            logger.warning("🚨 Using original features - CatBoost errors may occur")
            return X

    def diagnose_feature_issues(self, model, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced diagnostic for CatBoost feature compatibility
        """
        diagnosis = {
            'issues_found': [],
            'warnings': [],
            'recommendations': [],
            'feature_comparison': {}
        }
        
        try:
            # Get model's expected features
            model_features = None
            if hasattr(model, 'feature_names_'):
                model_features = model.feature_names_
            elif hasattr(model, 'get_feature_names'):
                model_features = model.get_feature_names()
            elif hasattr(model, 'feature_names'):
                model_features = model.feature_names
            
            if model_features is None:
                diagnosis['issues_found'].append("Cannot determine model's expected features")
                diagnosis['status'] = 'UNKNOWN'
                return diagnosis
            
            input_features = X.columns.tolist()
            
            diagnosis['feature_comparison'] = {
                'model_features_count': len(model_features),
                'input_features_count': len(input_features),
                'model_features_sample': model_features[:5],
                'input_features_sample': input_features[:5]
            }
            
            # Check feature existence
            missing_features = set(model_features) - set(input_features)
            extra_features = set(input_features) - set(model_features)
            
            if missing_features:
                diagnosis['issues_found'].append(f"Missing {len(missing_features)} features: {list(missing_features)}")
            
            if extra_features:
                diagnosis['warnings'].append(f"Extra {len(extra_features)} features: {list(extra_features)}")
            
            # Check feature order
            if list(model_features) != input_features:
                diagnosis['issues_found'].append("Feature order mismatch detected")
                
                # Find specific position mismatches
                for i, (model_feat, input_feat) in enumerate(zip(model_features[:10], input_features[:10])):
                    if model_feat != input_feat:
                        diagnosis['issues_found'].append(f"Position {i}: model expects '{model_feat}', found '{input_feat}'")
                        break
            
            # CatBoost specific checks
            model_name = type(model).__name__.lower()
            if 'catboost' in model_name:
                diagnosis['recommendations'].append("Use _align_features_with_model() for CatBoost feature alignment")
                
                # Check for categorical features
                try:
                    if hasattr(model, 'get_cat_feature_indices'):
                        cat_indices = model.get_cat_feature_indices()
                        if cat_indices:
                            cat_features = [model_features[i] for i in cat_indices if i < len(model_features)]
                            diagnosis['feature_comparison']['categorical_features'] = cat_features
                except:
                    pass
            
            if not diagnosis['issues_found']:
                diagnosis['status'] = 'OK'
                diagnosis['message'] = 'No feature issues detected'
            else:
                diagnosis['status'] = 'CRITICAL'
                # FIXED: Properly closed f-string
                diagnosis['message'] = f"Found {len(diagnosis['issues_found'])} critical feature issues"
            
            diagnosis['recommendations'].extend([
                "Ensure training and explanation data have identical feature sets",
                "Use the same preprocessing pipeline for training and explanation",
                "For CatBoost, save feature names during model training"
            ])
            
        except Exception as e:
            diagnosis['status'] = 'ERROR'
            diagnosis['error'] = str(e)
        
        return diagnosis

    def _get_shap_explainer(self, model, X: pd.DataFrame):
        """
        Enhanced SHAP explainer with robust CatBoost handling
        """
        model_name = type(model).__name__.lower()
        
        # Enhanced CatBoost handling
        if 'catboost' in model_name:
            try:
                logger.info("🐱 Using CatBoost-optimized SHAP explainer...")
                
                # CRITICAL: Align features before creating explainer
                X_aligned = self._align_features_with_model(model, X)
                
                # For CatBoost, use TreeExplainer with feature_perturbation='interventional'
                explainer = shap.TreeExplainer(
                    model, 
                    feature_perturbation='interventional',
                    model_output='raw'
                )
                
                logger.info("✅ CatBoost TreeExplainer created with feature alignment")
                return explainer
                
            except Exception as e:
                logger.error(f"❌ CatBoost TreeExplainer failed: {e}")
                logger.info("🔄 Falling back to KernelExplainer for CatBoost...")
                
                # Fallback: Use KernelExplainer with aligned features
                try:
                    X_aligned = self._align_features_with_model(model, X)
                    background_size = min(100, len(X_aligned))
                    background = shap.sample(X_aligned, background_size)
                    
                    # Create prediction function that handles feature alignment
                    def predict_function(X_array):
                        if hasattr(X_array, 'shape') and len(X_array.shape) == 1:
                            X_array = X_array.reshape(1, -1)
                        
                        # Convert to DataFrame with correct feature names
                        if hasattr(model, 'feature_names_'):
                            feature_names = model.feature_names_
                        else:
                            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
                        
                        X_df = pd.DataFrame(X_array, columns=feature_names)
                        return model.predict(X_df)
                    
                    explainer = shap.KernelExplainer(predict_function, background)
                    logger.info("✅ CatBoost KernelExplainer fallback successful")
                    return explainer
                    
                except Exception as fallback_error:
                    logger.error(f"❌ CatBoost KernelExplainer also failed: {fallback_error}")
                    raise
        
        # Tree-based models
        elif any(tree_type in model_name for tree_type in ['tree', 'forest', 'xgb', 'lightgbm']):
            try:
                return shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer failed, using KernelExplainer: {e}")
                background_size = min(50, len(X))
                background = shap.sample(X, background_size)
                return shap.KernelExplainer(model.predict, background)
        
        # Linear models
        elif any(linear_type in model_name for linear_type in ['linear', 'logistic', 'ridge', 'lasso']):
            return shap.LinearExplainer(model, X)
        
        # Neural networks and default
        else:
            background_size = min(50, len(X))
            background = shap.sample(X, background_size)
            return shap.KernelExplainer(model.predict, background)

    def create_comprehensive_explanation(self, 
                                       model, 
                                       X: pd.DataFrame, 
                                       y: Optional[pd.Series] = None,
                                       task_type: str = 'classification',
                                       sample_size: int = 100) -> Dict[str, Any]:
        """
        Create comprehensive model explanations with robust feature alignment
        """
        try:
            logger.info("🎯 Creating comprehensive model explanations...")
            
            # CRITICAL: Diagnose feature issues first
            diagnosis = self.diagnose_feature_issues(model, X)
            logger.info(f"🔍 Feature diagnosis: {diagnosis['status']}")
            
            if diagnosis['issues_found']:
                for issue in diagnosis['issues_found'][:3]:  # Show first 3 issues
                    logger.error(f"❌ Feature issue: {issue}")
            
            # Sample data for faster computation
            if len(X) > sample_size:
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices] if y is not None else None
            else:
                X_sample = X
                y_sample = y
            
            # CRITICAL: Align features for ALL explanation methods
            logger.info("🔄 Aligning features for all explanation methods...")
            X_sample_aligned = self._align_features_with_model(model, X_sample)
            logger.info(f"✅ Features aligned: {X_sample_aligned.shape}")
            
            explanations = {
                'global_explanations': {},
                'local_explanations': {},
                'feature_importance': {},
                'business_insights': [],
                'visualizations': {},
                'model_performance': {},
                'data_characteristics': {},
                'feature_diagnosis': diagnosis
            }
            
            # SHAP Explanations - WITH ROBUST FEATURE ALIGNMENT
            if SHAP_AVAILABLE:
                try:
                    logger.info("📊 Creating SHAP explanations with feature alignment...")
                    shap_results = self._create_shap_explanations(model, X_sample_aligned, task_type)
                    explanations['global_explanations']['shap'] = shap_results
                    explanations['visualizations']['shap'] = self._create_shap_visualizations(
                        shap_results, X_sample_aligned.columns
                    )
                    logger.info("✅ SHAP explanations completed successfully")
                except Exception as e:
                    logger.error(f"❌ SHAP explanation failed: {str(e)}")
                    explanations['global_explanations']['shap'] = {'error': f"SHAP failed: {str(e)}"}
            
            # LIME Explanations - WITH FEATURE ALIGNMENT
            if LIME_AVAILABLE:
                try:
                    logger.info("🍋 Creating LIME explanations with feature alignment...")
                    lime_results = self._create_lime_explanations(model, X_sample_aligned, task_type)
                    explanations['local_explanations']['lime'] = lime_results
                    explanations['visualizations']['lime'] = self._create_lime_visualizations(lime_results)
                    logger.info("✅ LIME explanations completed successfully")
                except Exception as e:
                    logger.error(f"❌ LIME explanation failed: {str(e)}")
                    explanations['local_explanations']['lime'] = {'error': f"LIME failed: {str(e)}"}
            
            # Partial Dependence Plots - WITH FEATURE ALIGNMENT
            if PDP_AVAILABLE and len(X_sample_aligned.columns) > 0:
                try:
                    pdp_results = self._create_partial_dependence_plots(model, X_sample_aligned, task_type)
                    explanations['global_explanations']['partial_dependence'] = pdp_results
                    explanations['visualizations']['partial_dependence'] = self._create_pdp_visualizations(pdp_results)
                except Exception as e:
                    logger.error(f"❌ PDP failed: {str(e)}")
            
            # Feature Importance (Model-based)
            try:
                model_importance = self._get_model_feature_importance(model, X_sample_aligned.columns)
                explanations['feature_importance']['model_based'] = model_importance
                explanations['visualizations']['feature_importance'] = self._create_feature_importance_plot(model_importance)
            except Exception as e:
                logger.error(f"❌ Feature importance failed: {str(e)}")
            
            # Generate business insights
            explanations['business_insights'] = self._generate_business_insights(
                explanations, X_sample_aligned.columns, task_type, model, X_sample_aligned, y_sample
            )
            
            # Model performance metrics - WITH ALIGNED FEATURES
            if y_sample is not None:
                try:
                    explanations['model_performance'] = self._calculate_model_performance(
                        model, X_sample_aligned, y_sample, task_type
                    )
                except Exception as e:
                    logger.error(f"❌ Model performance calculation failed: {str(e)}")
            
            # Data characteristics
            explanations['data_characteristics'] = self._analyze_data_characteristics(X_sample_aligned, y_sample)
            
            # Summary statistics
            explanations['summary'] = {
                'n_features': len(X_sample_aligned.columns),
                'n_samples': len(X_sample_aligned),
                'task_type': task_type,
                'explanation_methods': ['SHAP' if SHAP_AVAILABLE else None, 
                                      'LIME' if LIME_AVAILABLE else None,
                                      'PDP' if PDP_AVAILABLE else None],
                'most_important_features': self._get_top_features(explanations),
                'explanation_confidence': self._calculate_explanation_confidence(explanations),
                'feature_alignment_status': diagnosis.get('status', 'Unknown')
            }
            
            logger.info("✅ Comprehensive explanations created successfully")
            return explanations
            
        except Exception as e:
            logger.error(f"❌ Error creating comprehensive explanations: {str(e)}")
            return {'error': str(e)}

    def _create_shap_explanations(self, model, X: pd.DataFrame, task_type: str) -> Dict[str, Any]:
        """Create SHAP explanations with enhanced error handling"""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            logger.info("🔍 Calculating SHAP values...")
            
            # Get explainer with feature alignment
            explainer = self._get_shap_explainer(model, X)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values_main = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # Binary classification or regression
                shap_values_main = shap_values
            
            # Global feature importance
            feature_importance = np.abs(shap_values_main).mean(axis=0)
            feature_importance_dict = dict(zip(X.columns, feature_importance))
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), 
                                                key=lambda x: x[1], reverse=True))
            
            results = {
                'shap_values': shap_values_main,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                'feature_importance': feature_importance_dict,
                'feature_names_aligned': X.columns.tolist(),
                'summary_statistics': {
                    'mean_abs_shap': np.abs(shap_values_main).mean(),
                    'max_abs_shap': np.abs(shap_values_main).max(),
                    'top_features': list(feature_importance_dict.keys())[:5]
                }
            }
            
            logger.info("✅ SHAP values calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error creating SHAP explanations: {str(e)}")
            return {'error': str(e)}

    def _create_lime_explanations(self, model, X: pd.DataFrame, task_type: str) -> Dict[str, Any]:
        """Create LIME explanations with feature alignment"""
        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}
        
        try:
            logger.info("🍋 Creating LIME explanations...")
            
            # Create LIME explainer with aligned features
            explainer = LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),  # Use aligned feature names
                class_names=['Class 0', 'Class 1'] if task_type == 'classification' else None,
                mode=task_type,
                discretize_continuous=True,
                random_state=42
            )
            
            # Get explanations for multiple instances
            n_explanations = min(10, len(X))
            sample_indices = np.random.choice(len(X), n_explanations, replace=False)
            
            explanations = []
            for idx in sample_indices:
                try:
                    # Use aligned features for explanation
                    instance_values = X.iloc[idx].values
                    
                    instance_explanation = explainer.explain_instance(
                        instance_values,
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=min(10, len(X.columns))
                    )
                    
                    explanation_data = {
                        'instance_index': idx,
                        'feature_importance': dict(instance_explanation.as_list()),
                        'prediction_probability': instance_explanation.predict_proba if hasattr(instance_explanation, 'predict_proba') else None,
                        'intercept': instance_explanation.intercept[1] if hasattr(instance_explanation, 'intercept') else 0
                    }
                    
                    explanations.append(explanation_data)
                except Exception as e:
                    logger.warning(f"⚠️ LIME explanation failed for instance {idx}: {str(e)}")
                    continue
            
            # Aggregate feature importance across instances
            all_features = {}
            for exp in explanations:
                for feature, importance in exp['feature_importance'].items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(abs(importance))
            
            # Calculate average importance
            avg_importance = {}
            for feature, importances in all_features.items():
                if importances:
                    avg_importance[feature] = np.mean(importances)
            
            avg_importance = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
            
            results = {
                'instance_explanations': explanations,
                'average_feature_importance': avg_importance,
                'n_instances_explained': len(explanations),
                'top_features': list(avg_importance.keys())[:5] if avg_importance else []
            }
            
            logger.info("✅ LIME explanations completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error creating LIME explanations: {str(e)}")
            return {'error': str(e)}

    def _create_partial_dependence_plots(self, model, X: pd.DataFrame, task_type: str) -> Dict[str, Any]:
        """Create Partial Dependence Plots"""
        if not PDP_AVAILABLE:
            return {'error': 'Partial Dependence Plots not available'}
        
        try:
            logger.info("Creating Partial Dependence Plots...")
            
            pdp_results = {}
            features = list(range(min(3, len(X.columns))))
            
            for feature in features:
                try:
                    pdp, axes = partial_dependence(
                        model, X, [feature], 
                        kind='average',
                        grid_resolution=20
                    )
                    
                    feature_name = X.columns[feature]
                    pdp_results[feature_name] = {
                        'values': axes[0].tolist(),
                        'average': pdp[0].tolist(),
                        'feature_name': feature_name
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ PDP failed for feature {feature}: {str(e)}")
                    continue
            
            return pdp_results
            
        except Exception as e:
            logger.error(f"❌ Error creating Partial Dependence Plots: {str(e)}")
            return {'error': str(e)}

    def _get_model_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance directly from the model"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
            
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importances = np.abs(model.coef_)
                else:
                    importances = np.abs(model.coef_).mean(axis=0)
                importance_dict = dict(zip(feature_names, importances))
            
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"❌ Error getting model feature importance: {str(e)}")
            return {}

    def _generate_business_insights(self, 
                                  explanations: Dict[str, Any], 
                                  feature_names: List[str], 
                                  task_type: str,
                                  model=None,
                                  X: pd.DataFrame = None,
                                  y: pd.Series = None) -> List[str]:
        """Generate business-friendly insights from explanations"""
        insights = []
        
        try:
            top_features = set()
            
            if 'shap' in explanations.get('global_explanations', {}):
                shap_top = explanations['global_explanations']['shap'].get('summary_statistics', {}).get('top_features', [])
                top_features.update(shap_top[:3])
            
            if 'lime' in explanations.get('local_explanations', {}):
                lime_top = explanations['local_explanations']['lime'].get('top_features', [])
                top_features.update(lime_top[:3])
            
            if 'model_based' in explanations.get('feature_importance', {}):
                model_top = list(explanations['feature_importance']['model_based'].keys())[:3]
                top_features.update(model_top)
            
            top_features = list(top_features)[:5]
            
            if top_features:
                insights.append(f"The model's decisions are primarily driven by: {', '.join(top_features)}")
                
                if len(top_features) >= 1:
                    insights.append(f"**{top_features[0]}** has the strongest influence on predictions. Focus on data quality for this feature.")
                
                if len(top_features) >= 3:
                    insights.append(f"Consider interactions between **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}** for deeper business understanding.")
            
            if task_type == 'classification':
                insights.append("For classification tasks, features with higher positive SHAP values increase the probability of the positive class.")
                insights.append("Negative SHAP values decrease the probability of the positive class, indicating protective or reducing factors.")
            else:
                insights.append("For regression tasks, positive SHAP values increase the predicted target value.")
                insights.append("Negative SHAP values decrease the predicted target value, indicating inverse relationships.")
            
            if X is not None:
                missing_data = X.isnull().sum().sum()
                if missing_data > 0:
                    insights.append(f"⚠️ Dataset contains {missing_data} missing values. Consider imputation for better model performance.")
                
                if len(X.columns) > 1:
                    corr_matrix = X.corr()
                    high_corr = (corr_matrix.abs() > 0.8).sum().sum() - len(X.columns)
                    if high_corr > 0:
                        insights.append(f"Found {high_corr} highly correlated feature pairs. Consider feature selection to reduce multicollinearity.")
            
            if 'model_performance' in explanations:
                perf = explanations['model_performance']
                if task_type == 'classification' and 'accuracy' in perf:
                    accuracy = perf['accuracy']
                    if accuracy > 0.9:
                        insights.append("Excellent model accuracy achieved! The model is highly reliable for business decisions.")
                    elif accuracy > 0.7:
                        insights.append("Good model performance. Suitable for most business applications with proper validation.")
                    else:
                        insights.append("Model performance needs improvement. Consider feature engineering or trying different algorithms.")
            
            insights.extend([
                "Use these explanations to validate that the model aligns with business domain knowledge.",
                "Regularly monitor feature importance as it may change over time with new data.",
                "Consider A/B testing for top features to validate their business impact.",
                "Ensure data privacy and ethical considerations for sensitive features."
            ])
            
        except Exception as e:
            logger.error(f"❌ Error generating business insights: {str(e)}")
            insights.append("⚠️ Unable to generate detailed insights due to technical limitations")
            insights.append("Basic insight: Focus on model interpretability and business alignment")
        
        return insights

    def _calculate_model_performance(self, model, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, float]:
        """Calculate additional model performance metrics"""
        try:
            predictions = model.predict(X)
            
            if task_type == 'classification':
                return {
                    'accuracy': accuracy_score(y, predictions),
                    'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
                }
            else:
                return {
                    'r2': r2_score(y, predictions),
                    'mse': mean_squared_error(y, predictions),
                    'rmse': np.sqrt(mean_squared_error(y, predictions)),
                    'mae': mean_absolute_error(y, predictions)
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Model performance calculation failed: {str(e)}")
            return {}

    def _analyze_data_characteristics(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """Analyze data characteristics for better insights"""
        try:
            characteristics = {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_types': {
                    'numeric': len(X.select_dtypes(include=[np.number]).columns),
                    'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
                },
                'missing_values': X.isnull().sum().sum(),
                'data_quality': 'Good' if X.isnull().sum().sum() == 0 else 'Needs Attention'
            }
            
            if y is not None:
                characteristics['target_distribution'] = {
                    'n_unique': y.nunique(),
                    'balance': 'Balanced' if y.nunique() <= 10 and y.value_counts().std() / y.value_counts().mean() < 0.5 else 'Imbalanced'
                }
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"⚠️ Data characteristics analysis failed: {str(e)}")
            return {}

    def _get_top_features(self, explanations: Dict[str, Any], top_n: int = 5) -> List[str]:
        """Get top features across all explanation methods"""
        all_features = {}
        
        try:
            if 'shap' in explanations.get('global_explanations', {}):
                shap_features = explanations['global_explanations']['shap'].get('feature_importance', {})
                for feature, importance in shap_features.items():
                    all_features[feature] = all_features.get(feature, 0) + importance
            
            if 'lime' in explanations.get('local_explanations', {}):
                lime_features = explanations['local_explanations']['lime'].get('average_feature_importance', {})
                for feature, importance in lime_features.items():
                    clean_feature = feature.split()[0] if ' ' in feature else feature
                    all_features[clean_feature] = all_features.get(clean_feature, 0) + importance
            
            if 'model_based' in explanations.get('feature_importance', {}):
                model_features = explanations['feature_importance']['model_based']
                for feature, importance in model_features.items():
                    all_features[feature] = all_features.get(feature, 0) + importance
            
            sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
            return [feature for feature, _ in sorted_features[:top_n]]
            
        except Exception as e:
            logger.error(f"❌ Error getting top features: {str(e)}")
            return []

    def _calculate_explanation_confidence(self, explanations: Dict[str, Any]) -> str:
        """Calculate overall confidence in explanations"""
        try:
            confidence_indicators = []
            
            top_features_methods = []
            
            if 'shap' in explanations.get('global_explanations', {}):
                shap_top = explanations['global_explanations']['shap'].get('summary_statistics', {}).get('top_features', [])
                top_features_methods.append(set(shap_top[:3]))
            
            if 'lime' in explanations.get('local_explanations', {}):
                lime_top = explanations['local_explanations']['lime'].get('top_features', [])
                top_features_methods.append(set(lime_top[:3]))
            
            if 'model_based' in explanations.get('feature_importance', {}):
                model_top = list(explanations['feature_importance']['model_based'].keys())[:3]
                top_features_methods.append(set(model_top))
            
            if len(top_features_methods) >= 2:
                agreements = []
                for i in range(len(top_features_methods)):
                    for j in range(i+1, len(top_features_methods)):
                        intersection = len(top_features_methods[i].intersection(top_features_methods[j]))
                        agreements.append(intersection)
                
                avg_agreement = np.mean(agreements) if agreements else 0
                
                if avg_agreement >= 2:
                    confidence_indicators.append("high")
                elif avg_agreement >= 1:
                    confidence_indicators.append("medium")
                else:
                    confidence_indicators.append("low")
            else:
                confidence_indicators.append("medium")
            
            if explanations.get('data_characteristics', {}).get('data_quality') == 'Good':
                confidence_indicators.append("high")
            else:
                confidence_indicators.append("medium")
            
            if confidence_indicators.count("high") >= 2:
                return "High"
            elif confidence_indicators.count("medium") >= 1:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.warning(f"⚠️ Explanation confidence calculation failed: {str(e)}")
            return "Medium"

    def _create_shap_visualizations(self, shap_results: Dict[str, Any], feature_names: List[str]) -> Dict[str, go.Figure]:
        """Create SHAP visualizations"""
        visualizations = {}
        
        try:
            if 'feature_importance' in shap_results:
                importance_data = shap_results['feature_importance']
                top_features = dict(list(importance_data.items())[:15])
                
                fig = go.Figure(data=go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker_color='lightblue',
                    hovertemplate='<b>%{y}</b><br>SHAP Importance: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="SHAP Feature Importance - Global Explanations",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Features",
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                
                visualizations['feature_importance'] = fig
            
            if 'shap_values' in shap_results:
                shap_values = shap_results['shap_values']
                top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-10:]
                top_feature_names = [feature_names[i] for i in top_indices]
                
                fig = go.Figure()
                
                for i, feature_idx in enumerate(top_indices):
                    fig.add_trace(go.Box(
                        y=shap_values[:, feature_idx],
                        name=feature_names[feature_idx],
                        boxpoints='outliers',
                        marker_color='lightseagreen'
                    ))
                
                fig.update_layout(
                    title="SHAP Values Distribution - Top 10 Features",
                    yaxis_title="SHAP value (impact on model output)",
                    xaxis_title="Features",
                    height=500,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                
                visualizations['values_distribution'] = fig
            
        except Exception as e:
            logger.error(f"❌ Error creating SHAP visualizations: {str(e)}")
        
        return visualizations

    def _create_lime_visualizations(self, lime_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create LIME visualizations"""
        visualizations = {}
        
        try:
            if 'average_feature_importance' in lime_results:
                importance_data = lime_results['average_feature_importance']
                top_features = dict(list(importance_data.items())[:10])
                
                fig = go.Figure(data=go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker_color='lightcoral',
                    hovertemplate='<b>%{y}</b><br>LIME Importance: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="LIME Average Feature Importance - Local Explanations",
                    xaxis_title="Average |Importance|",
                    yaxis_title="Features",
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                
                visualizations['average_importance'] = fig
            
            if 'instance_explanations' in lime_results and lime_results['instance_explanations']:
                explanations = lime_results['instance_explanations']
                first_explanation = explanations[0]['feature_importance']
                
                features = list(first_explanation.keys())[:8]
                importances = [first_explanation[f] for f in features]
                
                colors = ['red' if imp < 0 else 'green' for imp in importances]
                
                fig = go.Figure(data=go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"LIME Explanation - Instance {explanations[0]['instance_index']}",
                    xaxis_title="Feature Contribution to Prediction",
                    yaxis_title="Features",
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                
                visualizations['instance_explanation'] = fig
            
        except Exception as e:
            logger.error(f"❌ Error creating LIME visualizations: {str(e)}")
        
        return visualizations

    def _create_pdp_visualizations(self, pdp_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create Partial Dependence Plot visualizations"""
        visualizations = {}
        
        try:
            for feature_name, pdp_data in pdp_results.items():
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=pdp_data['values'],
                    y=pdp_data['average'],
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6),
                    name='Partial Dependence'
                ))
                
                fig.update_layout(
                    title=f"Partial Dependence Plot - {feature_name}",
                    xaxis_title=feature_name,
                    yaxis_title="Partial Dependence",
                    height=400,
                    showlegend=True
                )
                
                visualizations[f'pdp_{feature_name}'] = fig
            
        except Exception as e:
            logger.error(f"❌ Error creating PDP visualizations: {str(e)}")
        
        return visualizations

    def _create_feature_importance_plot(self, importance_dict: Dict[str, float]) -> go.Figure:
        """Create feature importance plot"""
        try:
            if not importance_dict:
                return go.Figure().add_annotation(text="No feature importance data available")
            
            top_features = dict(list(importance_dict.items())[:15])
            
            fig = go.Figure(data=go.Bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                marker_color='lightseagreen',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Model-Based Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(top_features) * 25),
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Error creating feature importance plot: {str(e)}")
            return go.Figure()

    def explain_single_prediction(self, 
                                model, 
                                X_background: pd.DataFrame,
                                instance: pd.Series,
                                task_type: str = 'classification') -> Dict[str, Any]:
        """Explain a single prediction in detail with feature alignment"""
        try:
            logger.info("🔍 Explaining single prediction with feature alignment...")
            
            result = {
                'instance_data': instance.to_dict(),
                'prediction': None,
                'explanations': {},
                'confidence': {}
            }
            
            # CRITICAL: Align background and instance features
            X_background_aligned = self._align_features_with_model(model, X_background)
            instance_aligned = self._align_features_with_model(model, pd.DataFrame([instance]))
            
            # Get prediction with aligned features
            instance_array = instance_aligned.values.reshape(1, -1)
            prediction = model.predict(instance_array)[0]
            result['prediction'] = prediction
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(instance_array)[0]
                result['probabilities'] = probabilities
            
            # SHAP explanation with aligned features
            if SHAP_AVAILABLE:
                try:
                    explainer = self._get_shap_explainer(model, X_background_aligned)
                    shap_values = explainer.shap_values(instance_array)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    
                    shap_explanation = dict(zip(instance_aligned.columns, shap_values[0]))
                    result['explanations']['shap'] = shap_explanation
                    
                    total_effect = np.sum(np.abs(list(shap_explanation.values())))
                    if total_effect > 0.1:
                        result['confidence']['shap'] = 'High'
                    elif total_effect > 0.01:
                        result['confidence']['shap'] = 'Medium'
                    else:
                        result['confidence']['shap'] = 'Low'
                    
                except Exception as e:
                    logger.error(f"❌ Error in SHAP single prediction: {str(e)}")
                    result['explanations']['shap'] = {'error': str(e)}
            
            # LIME explanation with aligned features
            if LIME_AVAILABLE:
                try:
                    explainer = LimeTabularExplainer(
                        X_background_aligned.values,
                        feature_names=X_background_aligned.columns.tolist(),
                        mode=task_type,
                        random_state=42
                    )
                    
                    lime_exp = explainer.explain_instance(
                        instance_aligned.values[0],
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=len(instance_aligned.columns)
                    )
                    
                    result['explanations']['lime'] = dict(lime_exp.as_list())
                    result['confidence']['lime'] = 'High'
                    
                except Exception as e:
                    logger.error(f"❌ Error in LIME single prediction: {str(e)}")
                    result['explanations']['lime'] = {'error': str(e)}
            
            logger.info("✅ Single prediction explanation completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error explaining single prediction: {str(e)}")
            return {'error': str(e)}

    def create_explanation_report(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive explanation report"""
        try:
            report = {
                'summary': {
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'explanation_confidence': explanations.get('summary', {}).get('explanation_confidence', 'Unknown'),
                    'top_features': explanations.get('summary', {}).get('most_important_features', []),
                    'methods_used': [method for method in explanations.get('summary', {}).get('explanation_methods', []) if method],
                    'feature_alignment_status': explanations.get('feature_diagnosis', {}).get('status', 'Unknown')
                },
                'key_insights': explanations.get('business_insights', []),
                'model_performance': explanations.get('model_performance', {}),
                'data_characteristics': explanations.get('data_characteristics', {}),
                'feature_diagnosis': explanations.get('feature_diagnosis', {}),
                'recommendations': self._generate_recommendations(explanations)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error creating explanation report: {str(e)}")
            return {'error': str(e)}

    def _generate_recommendations(self, explanations: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on explanations"""
        recommendations = []
        
        try:
            data_char = explanations.get('data_characteristics', {})
            if data_char.get('missing_values', 0) > 0:
                recommendations.append("Address missing values through imputation or collection improvements")
            
            if data_char.get('feature_types', {}).get('categorical', 0) > 10:
                recommendations.append("Consider encoding strategies for high-cardinality categorical features")
            
            model_perf = explanations.get('model_performance', {})
            if model_perf.get('accuracy', 0) < 0.7:
                recommendations.append("Explore feature engineering or alternative algorithms to improve performance")
            
            top_features = explanations.get('summary', {}).get('most_important_features', [])
            if top_features:
                recommendations.append(f"Focus monitoring and data quality efforts on: {', '.join(top_features[:3])}")
            
            feature_diagnosis = explanations.get('feature_diagnosis', {})
            if feature_diagnosis.get('status') == 'CRITICAL':
                recommendations.append("🚨 RESOLVE FEATURE ALIGNMENT ISSUES for reliable explanations")
            
            recommendations.extend([
                "🔄 Establish regular model monitoring and retraining schedule",
                "📊 Implement model performance tracking with business metrics",
                "🔍 Conduct periodic feature importance analysis to detect concept drift",
                "✅ Validate model explanations with domain experts"
            ])
            
        except Exception as e:
            logger.warning(f"⚠️ Recommendation generation failed: {str(e)}")
            recommendations.append("Regularly review model performance and business alignment")
        
        return recommendations