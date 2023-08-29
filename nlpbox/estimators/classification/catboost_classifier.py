"""Esse módulo contém um classificador
do catboost.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from catboost import CatBoostClassifier as _CatBoostClassifier

from nlpbox.core.estimator import Estimator


class CatBoostClassifier(Estimator):
    def __init__(self,
                 iterations=None,
                 learning_rate=None,
                 depth=None,
                 l2_leaf_reg=None,
                 model_size_reg=None,
                 rsm=None,
                 loss_function=None,
                 border_count=None,
                 feature_border_type=None,
                 per_float_feature_quantization=None,
                 input_borders=None,
                 output_borders=None,
                 fold_permutation_block=None,
                 od_pval=None,
                 od_wait=None,
                 od_type=None,
                 nan_mode=None,
                 counter_calc_method=None,
                 leaf_estimation_iterations=None,
                 leaf_estimation_method=None,
                 random_seed=None,
                 use_best_model=None,
                 metric_period=None,
                 ctr_leaf_count_limit=None,
                 store_all_simple_ctr=None,
                 max_ctr_complexity=None,
                 has_time=None,
                 allow_const_label=None,
                 classes_count=None,
                 class_weights=None,
                 auto_class_weights=None,
                 one_hot_max_size=None,
                 random_strength=None,
                 name=None,
                 ignored_features=None,
                 custom_loss=None,
                 custom_metric=None,
                 eval_metric=None,
                 bagging_temperature=None,
                 fold_len_multiplier=None,
                 final_ctr_computation_mode=None,
                 approx_on_full_history=None,
                 boosting_type=None,
                 simple_ctr=None,
                 combinations_ctr=None,
                 per_feature_ctr=None,
                 bootstrap_type=None,
                 subsample=None,
                 sampling_unit=None,
                 dev_score_calc_obj_block_size=None,
                 max_depth=None,
                 n_estimators=None,
                 num_boost_round=None,
                 num_trees=None,
                 colsample_bylevel=None,
                 random_state=None,
                 reg_lambda=None,
                 objective=None,
                 eta=None,
                 max_bin=None,
                 scale_pos_weight=None,
                 early_stopping_rounds=None,
                 cat_features=None,
                 grow_policy=None,
                 min_data_in_leaf=None,
                 min_child_samples=None,
                 max_leaves=None,
                 num_leaves=None,
                 score_function=None,
                 leaf_estimation_backtracking=None,
                 ctr_history_unit=None,
                 monotone_constraints=None,
                 feature_weights=None,
                 penalties_coefficient=None,
                 first_feature_use_penalties=None,
                 model_shrink_rate=None,
                 model_shrink_mode=None,
                 langevin=None,
                 diffusion_temperature=None,
                 posterior_sampling=None,
                 boost_from_average=None,
                 text_features=None,
                 tokenizers=None,
                 dictionaries=None,
                 feature_calcers=None,
                 text_processing=None,
                 fixed_binary_splits=None
                 ):
        self._hyperparams = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            model_size_reg=model_size_reg,
            rsm=rsm,
            loss_function=loss_function,
            border_count=border_count,
            feature_border_type=feature_border_type,
            per_float_feature_quantization=per_float_feature_quantization,
            input_borders=input_borders,
            output_borders=output_borders,
            fold_permutation_block=fold_permutation_block,
            od_pval=od_pval,
            od_wait=od_wait,
            od_type=od_type,
            nan_mode=nan_mode,
            counter_calc_method=counter_calc_method,
            leaf_estimation_iterations=leaf_estimation_iterations,
            leaf_estimation_method=leaf_estimation_method,
            random_seed=random_seed,
            use_best_model=use_best_model,
            metric_period=metric_period,
            ctr_leaf_count_limit=ctr_leaf_count_limit,
            store_all_simple_ctr=store_all_simple_ctr,
            max_ctr_complexity=max_ctr_complexity,
            has_time=has_time,
            allow_const_label=allow_const_label,
            classes_count=classes_count,
            class_weights=class_weights,
            auto_class_weights=auto_class_weights,
            one_hot_max_size=one_hot_max_size,
            random_strength=random_strength,
            name=name,
            ignored_features=ignored_features,
            custom_loss=custom_loss,
            custom_metric=custom_metric,
            eval_metric=eval_metric,
            bagging_temperature=bagging_temperature,
            fold_len_multiplier=fold_len_multiplier,
            final_ctr_computation_mode=final_ctr_computation_mode,
            approx_on_full_history=approx_on_full_history,
            boosting_type=boosting_type,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            bootstrap_type=bootstrap_type,
            subsample=subsample,
            sampling_unit=sampling_unit,
            dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
            max_depth=max_depth,
            n_estimators=n_estimators,
            num_boost_round=num_boost_round,
            num_trees=num_trees,
            colsample_bylevel=colsample_bylevel,
            random_state=random_state,
            reg_lambda=reg_lambda,
            objective=objective,
            eta=eta,
            max_bin=max_bin,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=early_stopping_rounds,
            cat_features=cat_features,
            grow_policy=grow_policy,
            min_data_in_leaf=min_data_in_leaf,
            min_child_samples=min_child_samples,
            max_leaves=max_leaves,
            num_leaves=num_leaves,
            score_function=score_function,
            leaf_estimation_backtracking=leaf_estimation_backtracking,
            ctr_history_unit=ctr_history_unit,
            monotone_constraints=monotone_constraints,
            feature_weights=feature_weights,
            penalties_coefficient=penalties_coefficient,
            first_feature_use_penalties=first_feature_use_penalties,
            model_shrink_rate=model_shrink_rate,
            model_shrink_mode=model_shrink_mode,
            langevin=langevin,
            diffusion_temperature=diffusion_temperature,
            posterior_sampling=posterior_sampling,
            boost_from_average=boost_from_average,
            text_features=text_features,
            tokenizers=tokenizers,
            dictionaries=dictionaries,
            feature_calcers=feature_calcers,
            text_processing=text_processing,
            fixed_binary_splits=fixed_binary_splits)

        self._catboost_classifier = _CatBoostClassifier(**self._hyperparams)

    def predict(self, X: ArrayLike) -> np.ndarray:
        preds = self._catboost_classifier.predict(X)
        return np.array(preds)
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        self._catboost_classifier.fit(X, y)
    
    @property
    def hyperparameters(self) -> dict:
        return self._hyperparams
    
    @property
    def params(self) -> dict:
        params =  self._catboost_classifier.get_all_params()

        return {k: v for k, v in params.items()
                if k not in self.hyperparameters}
