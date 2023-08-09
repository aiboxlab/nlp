"""Esse móduglo contém características
relacionadas com ortografia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

from nlpbox import resources
from nlpbox.core import FeatureExtractor
from nlpbox.lazy_loading import lazy_import

from .utils import DataclassFeatureSet

langtool = lazy_import('language_tool_python')


@dataclass(frozen=True)
class OrtographyFeatures(DataclassFeatureSet):
    ortography_score: float



class OrthographyExtractor(FeatureExtractor):
    """ Classe que permite calcular uma nota/score para o aspecto
    ortografia. Exemplo de uso:

    >>> scorer = LangToolOrthographyScorer()
    >>> scorer('Texto do Estudante') # retorna um float
    """

    def __init__(self, tool: langtool.LanguageTool = None):
        if tool is None:
            tool = langtool.LanguageTool('pt-BR')

        def cleaner(text: str):
            text = text.strip()
            text = re.sub(r"[!\.,—]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text[0].upper() + text[1:]

        self._tool = tool
        self._tokenizer_pattern = re.compile(r'\s+')
        self._tokenizer = spacy.tokenizer.Tokenizer(
            spacy.blank('pt').vocab,
            token_match=self._tokenizer_pattern.match)
        self._cleaner = cleaner
        self._rules = {
            "Encontrado possível erro de ortografia.",
            "Palavras estrangeiras com diacríticos",
            "Uso de apóstrofe para palavras no plural",
            "Femininos irregulares",
            "Erro ortográfico: Abreviaturas da internet",
            "Palavras raras facilmente confundidas",
            "Palavras raras: Capitalização de nomes geográficos"
        }

    def extract(self, text: str) -> OrtographyFeatures:
        # Limpeza do texto (pontuação, whitespace, etc)
        text = self._cleaner(text)

        # Obter tokens (palavras)
        tokens = [token.text for token in self._tokenizer(text)]

        # Inicializando features
        score = 0.0

        # Calcular os erros ortográficos presentes no texto
        errors = self._check(text)
        score = 1.0 - (len(errors) / len(tokens))

        return OrtographyFeatures(ortography_score=score)

    def _check(self, text: str) -> List[Dict[str, str]]:
        # Realizar uma checagem no texto utilizando o LanguageTool
        matches = self._tool.check(text)

        # Lista dos erros
        errors = []

        for match in matches:
            # Caso sejam encontrados erros de ortografia:
            if match.message in self._rules:
                error_dict = {}
                correct_token = ''

                # Obter token/palavra original errôneo
                offset = match.offset
                length = match.errorLength
                token = text[offset: offset + length]

                # Buscar se existe um candidato à substituição
                if len(match.replacements) > 0:
                    correct_token = match.replacements[0]

                # Adicionar informações no dicionário para esse erro
                error_dict["token"] = token
                error_dict["correct_token"] = correct_token

                # Adicionar esse erro na lista
                errors.append(error_dict)

        return errors
