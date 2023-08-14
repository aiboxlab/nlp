"""Esse módulo contém uma classe
para experimentos simples.
"""
from __future__ import annotations

import numpy as np

from nlpbox.core import (Dataset, Experiment, ExperimentConfiguration,
                         ExperimentResult, Metric, Pipeline)
from nlpbox.lazy_loading import lazy_import

pandas_types = lazy_import('pandas.api.types')
feature_agg = lazy_import('nlpbox.features.utils.aggregator')


class SimpleExperiment(Experiment):
    def __init__(self,
                 pipelines: list[Pipeline],
                 dataset: Dataset,
                 metrics: list[Metric],
                 criteria_best: Metric,
                 pipelines_names: list[str] | None = None,
                 seed: int = 8990,
                 keep_all_pipelines: bool = False,
                 problem: str | None = None):
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
        """
        if problem is None:
            dtype = dataset.to_frame().target.dtype
            assert pandas_types.is_numeric_dtype(dtype)
            problem = 'classification' if pandas_types.is_integer_dtype(
                dtype) else 'regression'

        if pipelines_names is None:
            pipelines_names = list(map(self._generate_name, pipelines))

        assert len(pipelines) == len(pipelines_names)
        self._pipelines: list[tuple[str, Pipeline]] = list(zip(pipelines_names,
                                                               pipelines))
        self._seed = seed
        self._keep_all_pipelines = keep_all_pipelines
        self._dataset = dataset
        self._metrics = metrics
        self._best_criteria = criteria_best
        self._problem = problem
        self._validate()

    def run(self) -> ExperimentResult:
        """Executa esse experimento.

        Returns:
            ExperimentResult: resultado do experimento,
                essa classe não retorna nada na chave
                "extras".
        """
        best_pipeline = None
        best_name = None
        best_metrics = None
        best_test_predictions = None
        metrics_history = dict()
        pipeline_history = None
        rng = np.random.default_rng(self._seed)

        seed_splits = rng.integers(low=0, high=9999, endpoint=True)
        train, test = self._dataset.train_test_split(frac_train=0.8,
                                                     seed=seed_splits)
        X_train, y_train = train.text.to_numpy(), train.target.to_numpy()
        X_test, y_test = test.text.to_numpy(), test.target.to_numpy()

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

        for p in self._pipelines:
            name, pipeline = p

            # TODO: adicionar um replace nos
            #   feature extractors para ter
            #   cache durante treinamento.

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

        return ExperimentResult(best_pipeline=best_pipeline,
                                best_pipeline_name=best_name,
                                best_metrics=best_metrics,
                                best_test_predictions=None,
                                train_df=train,
                                test_df=test,
                                metrics_history=metrics_history,
                                pipeline_history=pipeline_history)

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
        metrics_names = list(m.name() for m in metrics)
        assert len(metrics_names) == len(set(metrics_names))

    def _generate_name(self, p: Pipeline) -> str:
        # Obtendo nome da classe do estimador
        estimator_name = p.estimator.__class__.__name__

        # Obtendo nome da classe do vetorizador
        vectorizer_name = p.vectorizer.__class__.__name__

        # Se for um agregado de features, obtemos o nome
        #   individual de cada um
        if isinstance(p.vectorizer, feature_agg.AggregatedFeatureExtractor):
            vectorizer_name = '_'.join(v.__class__.__name__
                                       for v in p.vectorizer.extractors)

        # Obtemos os parâmetros do estimador
        estimator_params = '_'.join(str(v) for v in
                                    p.estimator.hyperparameters.values()
                                    if not isinstance(v, dict))

        # Construímos o nome final da pipeline
        name = '_'.join([vectorizer_name, estimator_name, estimator_params])
        return name
