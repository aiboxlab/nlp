"""Esse método contém funções utilitárias
para construção e obtenção de experimentos
"""
from __future__ import annotations

import logging
from typing import Callable, ClassVar, Literal

import numpy as np

from aibox.nlp.core import Dataset, Experiment, Metric, Pipeline
from aibox.nlp.experiments.simple_experiment import SimpleExperiment
from aibox.nlp.pipelines import postprocessing as pp

from .class_registry import get_class, registry
from .feature_extractor import get_extractor

logger = logging.getLogger(__name__)


class SimpleExperimentBuilder:
    _SKIP_ESTIMATOR: ClassVar[set[str]] = {
        'lstmClf',
        'lstmReg'
    }

    def __init__(self) -> None:
        self._ds: Dataset = None
        self._criteria: Metric = None
        self._pipelines: list[Pipeline] = []
        self._metrics: list[Metric] = []
        self._seed = None
        self._rng = None
        self._problem = None

    def add_feature_pipeline(self,
                             features: str | list[str],
                             estimators: str | list[str],
                             names: str | list[str],
                             postprocessing: Callable | list[Callable] = None,
                             features_configs: dict | list[dict] = None,
                             estimators_configs: dict | list[dict] = None) -> SimpleExperimentBuilder:
        """Adiciona uma ou mais pipelines baseada em características. Se
        forem passados mais que um estimador, serão construídas pipelines
        com o mesmo conjunto de features mas com cada estimador.

        Args:
            features (str | list[str]): característica ou lista de características.
            estimator (str | list[str]): estimador ou lista de estimadores.
            features_configs (dict | list[dict], optional): configurações dos
                extratores de características. Defaults to None.

        Returns:
            SimpleExperimentBuilder: self.
        """
        if self._seed is None:
            logger.info('Inicialize a seed randômica primeiro.')
            return

        features = self._maybe_convert_to_list(features)
        estimators = self._maybe_convert_to_list(estimators)
        names = self._maybe_convert_to_list(names)

        if features_configs is None:
            features_configs = [dict()] * len(features)
        features_configs = self._maybe_convert_to_list(features_configs)

        if estimators_configs is None:
            estimators_configs = [dict()] * len(names)
        estimators_configs = self._maybe_convert_to_list(estimators_configs)

        if postprocessing is None:
            postprocessing = [None] * len(names)
        postprocessing = self._maybe_convert_to_list(postprocessing)

        assert len(estimators) == len(names)
        extractor = get_extractor(features,
                                  features_configs)
        for name, e, c, p in zip(names,
                                 estimators,
                                 estimators_configs,
                                 postprocessing):
            seed = self._estimator_seed()
            estimator = get_class(e)(**c, random_state=seed)
            self._pipelines.append(Pipeline(vectorizer=extractor,
                                            estimator=estimator,
                                            postprocessing=p,
                                            name=name))

        return self

    def add_vectorizer_pipeline(self,
                                vectorizer: str,
                                estimators: str | list[str],
                                names: str | list[str],
                                postprocessing: Callable | list[Callable] = None,
                                vectorizer_config: dict = dict(),
                                estimators_configs: dict | list[dict] = None) -> SimpleExperimentBuilder:
        """Adiciona uma ou mais pipelines baseadas no vetorizar. Se
        forem passados mais que um estimador, serão construídas pipelines
        com o mesmo vetorizador mas com cada estimador.

        Args:
            vectorizer (str): vetorizador ou lista de vetorizadores.
            estimators (str | list[str]): estimador ou lista de estimadores.
            names (str | list[str]): nome(s) do estimador(es).
            postprocessing (Callable | list[Callable], optional): pós-processamentos.
            vectorizer_config (dict, optional): configuração do vetorizador.
            estimators_configs (dict | list[dict], optional): configuração(ões) do estimador(es).

        Returns:
            SimpleExperimentBuilder: self.
        """
        if self._seed is None:
            logger.info('Inicialize a seed randômica primeiro.')
            return

        estimators = self._maybe_convert_to_list(estimators)
        names = self._maybe_convert_to_list(names)

        if estimators_configs is None:
            estimators_configs = [dict()] * len(names)
        estimators_configs = self._maybe_convert_to_list(estimators_configs)

        if postprocessing is None:
            postprocessing = [None] * len(names)
        postprocessing = self._maybe_convert_to_list(postprocessing)

        assert len(estimators) == len(names)
        vectorizer = get_class(vectorizer)(**vectorizer_config)

        for name, e, c, p in zip(names,
                                 estimators,
                                 estimators_configs,
                                 postprocessing):
            seed = self._estimator_seed()
            estimator = get_class(e)(**c, random_state=seed)
            self._pipelines.append(Pipeline(vectorizer=vectorizer,
                                            estimator=estimator,
                                            postprocessing=p,
                                            name=name))
        return self

    def add_metric(self,
                   metric: str,
                   **metric_config) -> SimpleExperimentBuilder:
        """Adiciona uma métrica para o experimento caso
        ela não tenha sido adicionada anteriormente.

        Args:
            metric (str): métrica.

        Returns:
            SimpleExperimentBuilder: self.
        """
        m = get_class(metric)(**metric_config)
        if m not in self._metrics:
            self._metrics.append(m)

        return self

    def best_criteria(self,
                      metric: str,
                      **metric_config) -> SimpleExperimentBuilder:
        """Define a métrica para selecionar
        a melhor pipeline.

        Args:
            metric (str): métrica.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._criteria = get_class(metric)(**metric_config)
        if self._criteria not in self._metrics:
            self._metrics.append(self._criteria)

        return self

    def dataset(self,
                ds: str,
                **ds_config) -> SimpleExperimentBuilder:
        """Define o dataset para
        os experimentos.

        Args:
            ds (str): dataset.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._ds = get_class(ds)(**ds_config)
        return self

    def seed(self, seed: int) -> SimpleExperimentBuilder:
        """Define a seed para o experimento.

        Args:
            seed (int): seed.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._seed = seed

        # Inicializando RNG para criar as seeds
        #   dos estimadores. Por garantia,
        #   utilizamos uma seed diferente da passada
        #   para o experimento.
        self._rng = np.random.default_rng(self._seed + 1)

        return self

    def classification(self) -> SimpleExperimentBuilder:
        """Define que esse é um experimento
        de classificação.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._problem = 'classification'
        return self

    def regression(self) -> SimpleExperimentBuilder:
        """Define que esse é um experimento
        de regressão.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._problem = 'regression'
        return self

    def custom_dataset(self,
                       ds: Dataset) -> SimpleExperimentBuilder:
        """Adiciona uma instância de um dataset.

        Args:
            ds (Dataset): dataset.

        Returns:
            SimpleExperimentBuilder: self.
        """
        self._ds = ds
        return self

    def build(self, **kwargs) -> Experiment:
        """Constrói o experimento com as informações
        coletadas e limpa as informações coletadas.

        Returns:
            Experiment: experimento.
        """
        # Construção do experimento
        experiment = SimpleExperiment(pipelines=self._pipelines,
                                      dataset=self._ds,
                                      criteria_best=self._criteria,
                                      metrics=self._metrics,
                                      seed=self._seed,
                                      keep_all_pipelines=False,
                                      problem=self._problem,
                                      **kwargs)

        # Reset do estado do builder
        self._ds: Dataset = None
        self._criteria: Metric = None
        self._pipelines: list[Pipeline] = []
        self._metrics: list[Metric] = []
        self._seed = None
        self._rng = None
        self._problem = None

        # Retornando o experimento
        return experiment

    @classmethod
    def features_experiment(cls,
                            seed: int,
                            problem: Literal['classification', 'regression'],
                            include_reg_as_clf: bool = True) -> SimpleExperimentBuilder:
        """Retorna uma instância pré-inicializada do builder com
        todas as pipelines utilizando todas características disponíveis.

        Args:
            seed (int): seed randômica.
            problem ('classification', 'regression'): tipo do problema.
            include_reg_as_clf (bool, optional): Se estimadores voltados
                à regressão devem ser adicionados para classificação
                utilizando `np.round`. (default=True)

        Returns:
            SimpleExperimentBuilder: builder.
        """
        # Inicializando builder
        builder = cls()
        builder.seed(seed)

        # Obtendo o nome de todas as características
        features = [k
                    for k in registry.features_br
                    if 'similarity' not in k.lower()]

        # Selecionando estimadores do tipo
        #   esperado (i.e., clf ou reg)
        target = f'{problem}.'
        estimators = [k
                      for k, v in registry.estimators.items()
                      if target in v and
                      k not in cls._SKIP_ESTIMATOR]
        names = [f'all_features+{e}'
                 for e in estimators]

        # Adicionando esses estimadores no experimento
        builder.add_feature_pipeline(features,
                                     estimators,
                                     names)

        # Atualizando problema do builder
        if problem == 'regression':
            # Adicionando métricas de regressão
            builder.add_metric('MAE')
            builder.add_metric('RMSE')
            builder.add_metric('R2')
            builder.add_metric('MSE')
            builder.regression()
        else:
            # Adicionando métricas de classificação
            builder.add_metric('precision')
            builder.add_metric('precision', average='weighted')
            builder.add_metric('recall')
            builder.add_metric('recall', average='weighted')
            builder.add_metric('f1')
            builder.add_metric('f1', average='weighted')
            builder.add_metric('kappa')
            builder.classification()

            # Adicionando regressores para
            #   classificação
            if include_reg_as_clf:
                estimators = [k
                              for k, v in registry.estimators.items()
                              if 'regression.' in v and
                              k not in cls._SKIP_ESTIMATOR]
                names = [f'all_features+{e}'
                         for e in estimators]
                postprocessing = [pp.round_to_integer]
                postprocessing = postprocessing * len(estimators)
                builder.add_feature_pipeline(features,
                                             estimators,
                                             names,
                                             postprocessing)

        return builder

    def _maybe_convert_to_list(self, obj) -> list:
        if not isinstance(obj, list):
            return [obj]

        return obj

    def _estimator_seed(self) -> int:
        return self._rng.integers(0, 99999)
