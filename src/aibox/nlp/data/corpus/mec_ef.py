"""Esse módulo contém o corpus
considerando as redações do Ensino
Fundamental do projeto do MEC.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Generator, Iterator

import pandas as pd

from aibox.nlp import resources


class Competence(Enum):
    COHESION = "Coesão"
    THEMATIC_COHERENCE = "Coerência Temática"
    FORMAL_REGISTER = "Registro Formal"
    TEXT_TYPOLOGY = "Tipologia Textual"

    @classmethod
    def from_str(cls, value: str) -> Competence:
        """Retorna a competência a partir de uma string.

        Args:
            value (str): valor.

        Returns:
            Competence: competência.
        """
        return next(filter(lambda o: o.name.lower() == value, Competence))


class AnnotationTopic(Enum):
    PLOT_EXPOSITION = "Orientação"
    PLOT_RISING_ACTION = "Complicação"
    PLOT_RESOLUTION = "Desfecho"
    NARRATIVE_NARRATOR = "Narrador"
    NARRATIVE_CHARACTER = "Personagem"
    NARRATIVE_TIME = "Organização temporal"
    NARRATIVE_SETTING = "Lugar/Espaço"
    NARRATIVE_ACTION = "Ação"
    PUNCTUATION_ERROR = "Erro de Pontuação"
    PUNCTUATION_COMMA_ERROR = "Erro de vírgula"
    ORTHOGRAPHY_ERROR = "grafia de palavras"
    ORTHOGRAPHY_SEGMENTATION = "Desvios de hipersegmentação/ hipossegmentação"
    GRAMMAR_PERSONAL_PRONOUN_ERROR = "Erro pronomes pessoais"
    GRAMMAR_ORALITY = "Presença de elementos da oralidade"
    GRAMMAR_AGREEMENT_ERROR = "Erro concordância nominal/verbal"
    GRAMMAR_REGENCY_ERROR = "Erro regência nominal/verbal"
    GRAMMAR_CONJUGATION_ERROR = "Incorreção na conjugação verbal"
    GRAMMAR_CONNECTIVES = "Erros de conectores ou de palavras de referência"
    SYNTAX_SENTENCE_ERROR = "Períodos compostos mal estruturados " "sintaticamente"
    SEMANTIC_WORD_INADEQUACY = "Palavra semanticamente inadequada"
    SEMANTIC_WORD_REPETITION = "Emprego repetitivo de palavras"
    COPY_PARAPHRASE = "Paráfrase texto motivador"
    COPY_FULL = "Cópia texto motivador"

    @classmethod
    def from_str(cls, value: str) -> AnnotationTopic:
        """Retorna o tópico a partir de uma string.

        Args:
            value (str): valor.

        Returns:
            AnnotationTopic: tópico.
        """
        return next(filter(lambda o: o.value == value, AnnotationTopic))


@dataclass(frozen=True)
class Annotation:
    topic: AnnotationTopic
    start_idx: int
    end_idx: int
    text_excerpt: str | None = None


@dataclass(frozen=True)
class AnnotatorData:
    competences: dict[Competence, int | None]
    annotations: list[Annotation]


@dataclass(frozen=True)
class Essay:
    text: str
    motivating_situation: str
    consolidated_competences: dict[Competence, int]
    annotator_1: AnnotatorData
    annotator_2: AnnotatorData


class CorpusMecEf:
    _KEY_MOTIV_SITUATION: ClassVar[str] = "motivating_situation"
    _KEY_TEXT: ClassVar[str] = "text"
    _KEY_COMPETENCES: ClassVar[str] = "consolidated_competences"
    _KEY_ANNOT1: ClassVar[str] = "annotator_1"
    _KEY_ANNOT2: ClassVar[str] = "annotator_2"
    _KEY_ANNOTATION: ClassVar[str] = "annotations"

    def __init__(self, load_excerpt: bool = True):
        """Construtor."""
        root_dir = resources.path("datasets/corpus-mec-ef.v1")
        json_path = root_dir.joinpath("dataset.json")
        with json_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        self._raw = json_data
        self._load_excerpt = load_excerpt

    def as_list(self) -> list[Essay]:
        return list(self.as_iterator())

    def as_generator(self) -> Generator[Essay]:
        def _generator():
            for entry in self._raw:
                yield self._essay_from_entry(entry)

        return _generator

    def as_iterator(self) -> Iterator[Essay]:
        return iter(self.as_generator()())

    def raw(self) -> list[dict]:
        """Obtém a representação cru
        desse corpus (JSON).

        Returns:
            list[dict]: lista de dicionários
                representando o corpus.
        """
        return copy.deepcopy(self._raw)

    def _essay_from_entry(self, entry: dict) -> Essay:
        consolidated = self._competences_dict_to_enum_dict(entry[self._KEY_COMPETENCES])
        text = entry[self._KEY_TEXT]
        motivating_situation = entry[self._KEY_MOTIV_SITUATION]
        annotations = {k: None for k in [self._KEY_ANNOT1, self._KEY_ANNOT2]}

        for k in annotations:
            # Acessando dados desse anotador
            annotator_entry = entry[k]

            # Obtendo competências desse anotador
            annotator_competences = {
                c: annotator_entry[c.name.lower()] for c in Competence
            }

            # Obtendo anotações a nível de caracteres
            annotations_entry = annotator_entry[self._KEY_ANNOTATION]
            annots = [self._annotation_from_entry(e, text) for e in annotations_entry]

            # Armazenando dados desse anotador
            annotations[k] = AnnotatorData(
                competences=annotator_competences, annotations=annots
            )

        return Essay(
            text=text,
            motivating_situation=motivating_situation,
            consolidated_competences=consolidated,
            annotator_1=annotations[self._KEY_ANNOT1],
            annotator_2=annotations[self._KEY_ANNOT2],
        )

    def _annotation_from_entry(self, entry: list, parent_text: str) -> Annotation:
        text = None
        start, end = entry[0], entry[1]
        topic = AnnotationTopic.from_str(entry[2])

        if self._load_excerpt:
            text = parent_text[start:end]

        return Annotation(topic=topic, start_idx=start, end_idx=end, text_excerpt=text)

    def _competences_dict_to_enum_dict(self, d: dict) -> dict[Competence, int]:
        return {Competence.from_str(k): v for k, v in d.items()}
