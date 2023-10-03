"""Esse módulo contém uma classe
para experimentos simples.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aibox.nlp.cache.features import CachedExtractor
from aibox.nlp.cache.mixed_feature_cache import MixedFeatureCache
from aibox.nlp.cache.vectorizers import (CachedVectorizer, DictVectorizerCache,
                                         TrainableCachedVectorizer)
from aibox.nlp.core import (Dataset, Experiment, ExperimentConfiguration,
                            ExperimentResult, FeatureExtractor, Metric,
                            Pipeline, TrainableVectorizer, Vectorizer)
from aibox.nlp.features.utils.aggregator import AggregatedFeatureExtractor
from aibox.nlp.lazy_loading import lazy_import

pandas_types = lazy_import('pandas.api.types')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimpleExperimentExtras:
    df_features: pd.DataFrame
    run_duration: float


class SimpleExperiment(Experiment):
    def __init__(self,
                 pipelines: list[Pipeline],
                 dataset: Dataset,
                 metrics: list[Metric],
                 criteria_best: Metric,
                 seed: int = 8990,
                 frac_train: float = 0.8,
                 keep_all_pipelines: bool = False,
                 cache_limit: int | None = 0,
                 problem: str | None = None,
                 features_df: pd.DataFrame | None = None,
                 stratified_ds: bool = True):
        """Classe para experimentos simples, onde
        as pipelines testadas são escolhidas pelo
        usuário e uma avaliação hold-out é
        utilizada.

        Essa classe implementa um sistema de 
        cacheamento de vetorização, permitindo o
        aproveitamento das saídas de um vetorizador.

        Args:
            pipelines (list[Pipeline]): pipelines que devem
                ser testadas.
            dataset (Dataset): que deve ser utilizado.
            metrics (list[Metric): métricas que devem ser
                calculadas.
            criteria_best (Metric): métrica que é utilizada
                para escolher a melhor pipeline.
            seed (int): seed randômica utilizada (default=8990).
            keep_all_pipelines (bool): se todas pipelines
                devem ser guardadas (default=False).
            cache_limit (int): permite realizar o cacheamento
                da vetorização entre as diferentes pipelines. Se
                <= 0, todos os textos são cacheados. Se >0, apenas
                `cache_limit` textos serão cacheados. Se for None,
                nenhum cacheamento é aplicado. (default=0)
            problem (str): 'classification', 'regression' ou
                None. Caso None, inferir do dataset (default=None).
            features_df (pd.DataFrame): DataFrame com características
                pré-extraídas ou None (default=None). O DataFrame precisa
                ter uma coluna 'text' e todas as demais colunas são
                relativas à uma característica existente na biblioteca.
            stratified_ds (bool): se devemos utilizar splits de train
                e test estratificados (default=True).
        """
        if problem is None:
            dtype = dataset.to_frame().target.dtype
            assert pandas_types.is_numeric_dtype(dtype)
            problem = 'classification' if pandas_types.is_integer_dtype(
                dtype) else 'regression'

        def _pipeline_priority(p: Pipeline) -> int:
            vectorizer = p.vectorizer

            if isinstance(vectorizer, AggregatedFeatureExtractor):
                return len(vectorizer.extractors)

            return 1

        # Obtendo as caches iniciais
        initial_features = self._df_to_dict(features_df)

        # Instanciando e fazendo sort das pipelines
        self._pipelines = pipelines
        self._pipelines = sorted(self._pipelines,
                                 key=_pipeline_priority)

        # Variáveis auxiliares
        self._seed = seed
        self._keep_all_pipelines = keep_all_pipelines
        self._dataset = dataset
        self._metrics = metrics
        self._best_criteria = criteria_best
        self._problem = problem
        self._cache_limit = cache_limit
        self._frac_train = frac_train
        self._feature_cache = MixedFeatureCache(target_features=None,
                                                initial_cache=initial_features,
                                                max_limit=self._cache_limit)
        self._generic_cache = dict()
        self._stratified = stratified_ds
        self._validate()

    def run(self) -> ExperimentResult:
        """Executa esse experimento.

        Returns:
            ExperimentResult: resultado do experimento,
                essa classe não retorna nada na chave
                "extras".
        """
        logger.info('Setting up experiment...')
        best_pipeline: Pipeline = None
        best_metrics = None
        best_test_predictions = None
        metrics_history = dict()
        pipeline_history = dict()
        rng = np.random.default_rng(self._seed)

        logger.info('Obtaining train and test split...')
        seed_splits = rng.integers(low=0, high=9999, endpoint=True)
        train, test = self._dataset.train_test_split(
            frac_train=self._frac_train,
            seed=seed_splits,
            stratified=self._stratified)
        X_train, y_train = train.text.to_numpy(), train.target.to_numpy()
        X_test, y_test = test.text.to_numpy(), test.target.to_numpy()
        logger.info('Train has %d samples, Test has %d samples.',
                    len(train), len(test))

        def _update_best(pipeline,
                         metrics,
                         predictions):
            nonlocal best_pipeline
            nonlocal best_metrics
            nonlocal best_test_predictions
            best_pipeline = pipeline
            best_metrics = metrics
            best_test_predictions = predictions

        logger.info('Run started.')
        run_start = time.perf_counter()
        i = 0
        n_pipelines = len(self._pipelines)

        while self._pipelines:
            i += 1

            # Obtendo pipeline
            pipeline = self._pipelines.pop()
            name = pipeline.name
            logger.info('Started pipeline "%s" (%d/%d)',
                        name, i, n_pipelines)

            # Obtendo pipeline a ser treinada
            maybe_cached_pipeline = self._maybe_cached_pipeline(pipeline)

            # Treinamento da pipeline
            maybe_cached_pipeline.fit(X_train, y_train)

            # Predições
            predictions = maybe_cached_pipeline.predict(X_test)

            # Cálculo das métricas
            metrics_result = {
                m.name(): m.compute(y_true=y_test,
                                    y_pred=predictions)
                for m in self._metrics
            }

            # Calculando melhor pipeline
            criteria = self._best_criteria.name()
            if best_pipeline is None or \
                    metrics_result[criteria] > best_metrics[criteria]:
                _update_best(pipeline,
                             metrics_result,
                             predictions)

            # Armazenando resultados das métricas
            #   no histórico.
            metrics_history[name] = metrics_result

            # Caso a pipeline deva ser guardada no
            #   histórico, salvamos ela.
            if self._keep_all_pipelines:
                pipeline_history[name] = pipeline

        run_duration = time.perf_counter() - run_start
        logger.info('Run finished in %.2f seconds.\n'
                    'Best pipeline: %s',
                    run_duration,
                    best_pipeline.name)

        # Obtendo DataFrame das features
        #   extraídas.
        df_features = self._features_df()

        # Criando o objeto usado em "extras"
        extras = SimpleExperimentExtras(df_features=df_features,
                                        run_duration=run_duration)

        # Retornando resultados
        return ExperimentResult(
            best_pipeline=best_pipeline,
            best_metrics=best_metrics,
            best_pipeline_test_predictions=best_test_predictions,
            train_df=train,
            test_df=test,
            metrics_history=metrics_history,
            pipeline_history=pipeline_history,
            extras=extras)

    def config(self) -> ExperimentConfiguration:
        return ExperimentConfiguration(
            dataset=self._dataset,
            metrics=self._metrics,
            best_criteria=self._best_criteria,
            extras=dict(problem=self._problem,
                        keep_all_pipelines=self._keep_all_pipelines,
                        fraction_train=self._frac_train,
                        stratified_ds=self._stratified,
                        cache_limit=self._cache_limit))

    def _validate(self):
        """Realiza uma validação nos
        componentes da classe.
        """
        # Não podem existir pipelines duplicadas
        names = [p.name for p in self._pipelines]
        assert len(names) == len(set(names))

        # Não podem existir métricas duplicadas
        metrics_names = list(m.name() for m in self._metrics)
        assert len(metrics_names) == len(set(metrics_names))

    def _maybe_cached_pipeline(self, pipeline: Pipeline) -> Pipeline:
        # Caso seja um extrator de características,
        #   atualizamos para utilizar uma versão cacheada.
        if isinstance(pipeline.vectorizer, FeatureExtractor):
            # Coletando informações sobre quais features são
            #   extraídas pelo vetorizador
            sample_features = pipeline.vectorizer.extract(
                'Texto de exemplo.')

            # Atualizar a memória para retornar
            #   apenas essas características.
            self._feature_cache.target_features = set(
                sample_features.as_dict().keys())

            # Instanciando um novo extrator com cache.
            cached_extractor = CachedExtractor(pipeline.vectorizer,
                                               self._feature_cache)

            # Nova pipeline que compartilha o mesmo estimador,
            #   vetorizador e pós-processamento que a original.
            return Pipeline(vectorizer=cached_extractor,
                            estimator=pipeline.estimator,
                            postprocessing=pipeline.postprocessing,
                            name=pipeline.name)

        # Do contrário, retornamos a pipeline
        #   com um cache genérico.
        target_cls = CachedVectorizer
        if isinstance(pipeline.vectorizer, TrainableVectorizer):
            target_cls = TrainableCachedVectorizer

        vec_id = self._vectorizer_id(pipeline.vectorizer)
        if vec_id not in self._generic_cache:
            self._generic_cache[vec_id] = DictVectorizerCache()

        cached_vectorizer = target_cls(vectorizer=pipeline.vectorizer,
                                       memory=self._generic_cache[vec_id])

        return Pipeline(vectorizer=cached_vectorizer,
                        estimator=pipeline.estimator,
                        postprocessing=pipeline.postprocessing,
                        name=pipeline.name)

    def _features_df(self) -> pd.DataFrame | None:
        # Inicializando variável
        df = None

        # Construindo os dados de features que foram
        #   cacheados.
        features_data = []
        memory_dict = self._feature_cache.as_dict()
        for k, v in memory_dict.items():
            features_data.append({
                'text': k,
                **v.as_dict()
            })

        # Caso existam dados
        if features_data:
            # Construindo DataFrame
            df = pd.DataFrame(features_data)

            # Removendo colunas que não possuem valor
            #   para todos os textos
            df = df.dropna(axis=1)

        return df

    def _vectorizer_id(self, v: Vectorizer) -> str:
        cls_name = v.__class__.__name__
        return f'{cls_name}: {id(v)}'

    def _df_to_dict(self, df: pd.DataFrame | None) -> dict:
        if df is not None:
            keys = df.text.to_list()
            values = df.drop(columns='text').to_dict(orient='records')
            return dict(zip(keys, values))

        return dict()
