"""Esse módulo contém a implementação
de um Class Registry, permitindo que classes
sejam armazenados como uma string.
"""
from __future__ import annotations

import importlib

_registry_features = {
    'agreementBR': 'agreement.AgreementExtractor',
    'bertSimilarityBR': 'bert_similarity.BERTSimilarityExtractor',
    'cohmetrixBR': 'cohmetrix.CohMetrixExtractor',
    'conjugationBR': 'conjugation.ConjugationExtractor',
    'connectivesV1BR': 'connectives_v1.ConnectivesExtractorV1',
    'connectivesV2BR': 'connectives_v2.ConnectivesExtractorV2',
    'descriptiveBR': 'descriptive.DescriptiveExtractor',
    'fuzzySimilarity': ('fuzzy_search_similarity.'
                        'FuzzySearchSimilarityExtractor'),
    'lexicalDiversityBR': 'lexical_diversity.LexicalDiversityExtractor',
    'liwcBR': 'liwc.LiwcExtractor',
    'nilcSimilarityBR': 'nilc_similarity.NILCSimilarityExtractor',
    'orthographyBR': 'orthography.OrthographyExtractor',
    'overlapBR': 'overlap.OverlapExtractor',
    'readabilityBR': 'readability.ReadabilityExtractor',
    'regencyBR': 'regency.RegencyExtractor',
    'semanticCohesionTransformersBR': ('semantic_cohesion_transformers.'
                                       'SemanticExtractorTransformers'),
    'semanticCohesionBR': 'semantic_cohesion.SemanticExtractor',
    'sequentialCohesionBR': ('sequential_cohesion.Sequential'
                             'CohesionExtractor'),
    'syntacticComplexityBR': ('syntactic_complexity.'
                              'SyntacticComplexityExtractor'),
    'textualSimplicityBR': 'textual_simplicity.TextualSimplicityExtractor',
    'tfidfSimilarity': 'tfidf_similarity.TFIDFSimilarityExtractor',
    'wordSegmentationBR': 'word_segmentation.WordSegmentationExtractor',
}

_registry_vectorizers = {
    'tfidfVectorizer': 'tfidf_vectorizer.TFIDFVectorizer',
    'bertVectorizer': 'bert_vectorizer.BertVectorizer',
    'fasttextWordVectorizer': ('fasttext_word_vectorizer'
                               '.FasttextWordVectorizer'),
}

_registry_metrics = {
    'R2': 'errors.R2',
    'MAE': 'errors.MAE',
    'MSE': 'errors.MSE',
    'RMSE': 'errors.RMSE',
    'kappa': 'kappa.CohensKappaScore',
    'neighborKappa': 'kappa.NeighborCohensKappaScore',
    'precision': 'precision.Precision',
    'recall': 'recall.Recall',
    'f1': 'f1_score.F1Score'
}

_registry_estimators = {
    'svm': 'classification.svm.SVM',
    'catboostClf': 'classification.catboost_classifier.CatBoostClassifier',
    'catboosetReg': 'regression.catboost_regressor.CatBoostRegressor',
    'etreesClf': 'classification.extra_trees_classifier.ExtraTreesClassifier',
    'etreesReg': 'regression.extra_trees_regressor.ExtraTreesRegressor',
    'lgbmClf': 'classification.lgbm_classifier.LGBMClassifier',
    'lgbmReg': 'regression.lgbm_regressor.LGBMRegressor',
    'lstmClf': 'classification.lstm_classifier.LSTMClassifier',
    'lstmReg': 'regression.lstm_regressor.LSTMClassifier',
    'rfClf': 'classification.random_forest_classifier.RandomForestClassifier',
    'rfReg': 'regression.random_forest_regressor.RandomForestRegressor',
    'xgbClf': 'classification.xgboost_classifier.XGBoostClassifier',
    'xgbReg': 'regression.xgboost_regressor.XGBoostRegressor',

}

_registry_datasets = {
    'essayBR': 'essay_br.DatasetEssayBR',
    'efBR': 'mec_ef.DatasetMecEf'
}

_registry: dict[str, str] = dict()
_registry.update({
    k: f'features.{v}'
    for k, v in _registry_features.items()
})
_registry.update({
    k: f'estimators.{v}'
    for k, v in _registry_estimators.items()
})
_registry.update({
    k: f'data.datasets.{v}'
    for k, v in _registry_datasets.items()
})
_registry.update({
    k: f'metrics.{v}'
    for k, v in _registry_metrics.items()
})
_registry.update({
    k: f'vectorizers.{v}'
    for k, v in _registry_vectorizers.items()
})


def get_class(key: str) -> type:
    """Retorna a classe do identificador
    recebido como argumento.

    Args:
        key (str): identificador.

    Returns:
        Classe.
    """
    # Obtendo nome do pacote e classe
    class_path = _registry[key]
    splits = class_path.rsplit('.', 1)
    module_name = f'nlpbox.{splits[0]}'
    class_name = splits[1]

    # Carregando módulo
    module = importlib.import_module(module_name)

    # Obtendo classe dentro desse módulo
    cls = getattr(module, class_name)

    return cls
