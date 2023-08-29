"""Esse módulo contém a implementação
de um Class Registry, permitindo que classes
sejam armazenados como uma string.
"""
from __future__ import annotations

import importlib

_registry_features = {
    'agreementBR': 'agreement.AgreementExtractor',
    'cohmetrixBR': 'cohmetrix.CohMetrixExtractor',
    'bertSimilarityBR': 'bert_similarity.BERTSimilarityExtractor',
    'conjugationBR': 'conjugation.ConjugationExtractor',
    'connectivesV1BR': 'connectives_v1.ConnectivesExtractorV1',
    'connectivesV2BR': 'connectives_v2.ConnectivesExtractorV2',
    'fuzzySimilarity': ('fuzzy_search_similarity.'
                        'FuzzySearchSimilarityExtractor'),
    'liwcBR': 'liwc.LiwcExtractor',
    'nilcSimilarityBR': 'nilc_similarity.NILCSimilarityExtractor',
    'orthographyBR': 'orthography.OrthographyExtractor',
    'overlapBR': 'overlap.OverlapExtractor',
    'regencyBR': 'regency.RegencyExtractor',
    'semanticCohesionTransBR': ('semantic_cohesion_transformers.'
                                'SemanticExtractorTransformers'),
    'tfidfSimilarity': 'tfidf_similarity.TFIDFSimilarityExtractor',
    'wordSegmentationBR': 'word_segmentation.WordSegmentationExtractor',
    'lexicalDiversityBR': 'lexical_diversity.LexicalDiversityExtractor',
}

_registry_vectorizers = {
    'tfidfVectorizer': 'tfidf_vectorizer.TFIDFVectorizer'
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
    'svm': 'classification.svm.SVM'
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
