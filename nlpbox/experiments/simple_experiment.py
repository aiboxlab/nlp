"""Esse módulo contém uma classe
para experimentos simples.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nlpbox.core import (Dataset, Experiment, ExperimentConfiguration,
                         ExperimentResult, FeatureExtractor, Metric, Pipeline)
from nlpbox.experiments.cache.mixed_feature_cache import MixedFeatureCache
from nlpbox.features.utils.aggregator import AggregatedFeatureExtractor
from nlpbox.features.utils.cache import CachedExtractor
from nlpbox.lazy_loading import lazy_import

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
                 pipelines_names: list[str] | None = None,
                 seed: int = 8990,
                 keep_all_pipelines: bool = False,
                 problem: str | None = None,
                 features_df: pd.DataFrame | None = None,
                 stratified_ds: bool = True):
        """Classe para experimentos simples, onde
        as pipelines testadas são escolhidas pelo
        usuário e hold-out evaluation (80/20) é
        utilizado.

        Args:
            pipelines (list[Pipeline]): pipelines que devem
                ser testadas.
            dataset (Dataset): que deve ser utilizado.
            metrics (list[Metric): métricas que devem ser
                calculadas.
            criteria_best (Metric): métrica que é utilizada
                para escolher a melhor pipeline.
            pipelines_names (list[str]): nome das pipelines ou None.
                Caso None, definir nomes automaticamente (default=None).
            seed (int): seed randômica utilizada (default=8990).
            keep_all_pipelines (bool): se todas pipelines
                devem ser guardadas (default=False).
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

        if pipelines_names is None:
            pipelines_names = list(map(self._generate_name, pipelines))

        initial_cache = dict()
        if features_df is not None:
            keys = features_df.text.to_list()
            values = features_df.drop(columns='text').to_dict(orient='records')
            initial_cache = dict(zip(keys, values))

        def _pipeline_priority(p) -> int:
            _, pipeline = p
            vectorizer = pipeline.vectorizer

            if isinstance(vectorizer, AggregatedFeatureExtractor):
                return len(vectorizer.extractors)

            return 1

        # Instanciando e fazendo sort das pipelines
        assert len(pipelines) == len(pipelines_names)
        self._pipelines = list(zip(pipelines_names, pipelines))
        self._pipelines = sorted(self._pipelines,
                                 key=_pipeline_priority,
                                 reverse=True)

        self._seed = seed
        self._keep_all_pipelines = keep_all_pipelines
        self._dataset = dataset
        self._metrics = metrics
        self._best_criteria = criteria_best
        self._problem = problem
        self._feature_cache = MixedFeatureCache(target_features=None,
                                                initial_cache=initial_cache)
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
        best_pipeline = None
        best_name = None
        best_metrics = None
        best_test_predictions = None
        metrics_history = dict()
        pipeline_history = None
        rng = np.random.default_rng(self._seed)

        logger.info('Obtaining train and test split...')
        seed_splits = rng.integers(low=0, high=9999, endpoint=True)
        train, test = self._dataset.train_test_split(
            frac_train=0.8,
            seed=seed_splits,
            stratified=self._stratified)
        X_train, y_train = train.text.to_numpy(), train.target.to_numpy()
        X_test, y_test = test.text.to_numpy(), test.target.to_numpy()
        logger.info('Train has %d samples, Test has %d samples.',
                    len(train), len(test))

        def _update_best(pipeline,
                         metrics,
                         pipeline_name,
                         predictions):
            nonlocal best_pipeline
            nonlocal best_metrics
            nonlocal best_name
            nonlocal best_test_predictions
            best_pipeline = pipeline
            best_metrics = metrics
            best_name = pipeline_name
            best_test_predictions = predictions

        logger.info('Run started.')
        run_start = time.perf_counter()
        i = 0

        for p in self._pipelines:
            name, pipeline = p
            i += 1
            logger.info('Started pipeline "%s" (%d/%d)',
                        name, i, len(self._pipelines))

            # Caso seja um extrator de características,
            #   atualizamos para utilizar uma versão cacheada.
            if isinstance(pipeline.vectorizer, FeatureExtractor):
                # Colentado informações sobre quais features são
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

                # Atualizando a variável local para apontar para uma
                #   nova pipeline que **compartilha** o mesmo estimador,
                #   vetorizador e pós-processamento que a originak.
                pipeline = Pipeline(vectorizer=cached_extractor,
                                    estimator=pipeline.estimator,
                                    postprocessing=pipeline.postprocessing)

            # Treinamento da pipeline
            pipeline.fit(X_train, y_train)

            # Predições
            preds = pipeline.predict(X_test)

            # Cálculo das métricas
            r = {m.name(): m.compute(y_true=y_test,
                                     y_pred=preds)
                 for m in self._metrics}

            # Calculando melhor pipeline
            c = self._best_criteria.name()
            if best_pipeline is None or r[c] > best_metrics[c]:
                _update_best(pipeline,
                             r,
                             name,
                             preds)

            # Armazenando no histórico
            metrics_history[name] = r

            if self._keep_all_pipelines:
                if pipeline_history is None:
                    pipeline_history = dict()
                pipeline_history[name] = pipeline

        run_duration = time.perf_counter() - run_start
        logger.info('Run finished in %.2f seconds.\n'
                    'Best pipeline: %s',
                    run_duration,
                    best_name)

        # Construindo os dados de features que foram
        #   cacheados.
        features_data = dict(text=[])
        memory_dict = self._feature_cache.as_dict()
        for i, (k, v) in enumerate(memory_dict.items()):
            v = v.as_dict()

            if i == 0:
                features_data['text'] = [k]
                for k1, v1 in v.items():
                    features_data[k1] = [v1]
            else:
                features_data['text'].append(k)
                for k1, v1 in v.items():
                    features_data[k1].append(v1)

        # Construindo DataFrame
        df_features = pd.DataFrame(features_data)
        assert df_features.notna().all().all()

        # Criando o objeto usado em "extras"
        extras = SimpleExperimentExtras(df_features=df_features,
                                        run_duration=run_duration)

        return ExperimentResult(
            best_pipeline=best_pipeline,
            best_pipeline_name=best_name,
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
                        keep_all_pipelines=self._keep_all_pipelines))

    def _validate(self):
        """Realiza uma validação nos
        componentes da classe.
        """
        # Não podem existir pipelines duplicadas
        names = [name for name, _ in self._pipelines]
        assert len(names) == len(set(names))

        # Não podem existir métricas duplicadas
        metrics_names = list(m.name() for m in self._metrics)
        assert len(metrics_names) == len(set(metrics_names))

    def _generate_name(self, p: Pipeline) -> str:
        # Obtendo nome da classe do estimador
        estimator_name = p.estimator.__class__.__name__

        # Obtendo nome da classe do vetorizador
        vectorizer_name = p.vectorizer.__class__.__name__

        # Se for um agregado de features, obtemos o nome
        #   individual de cada um
        if isinstance(p.vectorizer, AggregatedFeatureExtractor):
            vectorizer_name = '_'.join(v.__class__.__name__
                                       for v in p.vectorizer.extractors)

        # Obtemos os parâmetros do estimador
        estimator_params = '_'.join(str(v) for v in
                                    p.estimator.hyperparameters.values()
                                    if not isinstance(v, dict))

        # Construímos o nome final da pipeline
        name = '_'.join([vectorizer_name, estimator_name, estimator_params])
        return name
