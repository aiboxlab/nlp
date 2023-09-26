"""Esse módulo contém um wrapper
para as características do CohMetrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from nlpbox.core.feature_extractor import FeatureExtractor
from nlpbox.lazy_loading import lazy_import

from .utils import DataclassFeatureSet

cohmetrixBR = lazy_import('cohmetrixBR.features')


@dataclass(frozen=True)
class CohMetrixFeatures(DataclassFeatureSet):
    """Essa classe possui todas as características
    disponibilizadas pelo CohMetrix BR.
    """
    despc: float
    despc2: float
    despl: float
    despld: float
    dessc: float
    dessl: float
    dessld: float
    deswc: float
    deswlsy: float
    deswlsyd: float
    deswllt: float
    deswlltd: float
    crfno1: float
    crfao1: float
    crfso1: float
    crfnoa: float
    crfaoa: float
    crfsoa: float
    crfcwo1: float
    crfcwo1d: float
    crfcwoa: float
    crfcwoad: float
    ldttrc: float
    ldttra: float
    ldmtlda: float
    ldvocda: float
    cncadc: float
    cncadd: float
    cncall: float
    cncalter: float
    cnccaus: float
    cnccomp: float
    cncconce: float
    cncconclu: float
    cnccondi: float
    cncconfor: float
    cncconse: float
    cncexpli: float
    cncfinal: float
    cncinte: float
    cnclogic: float
    cncneg: float
    cncpos: float
    cncprop: float
    cnctemp: float
    smintep: float
    smintep_sentence: float
    sminter: float
    smcauswn: float
    synle: float
    synnp: float
    synmedpos: float
    synmedlem: float
    synmedwrd: float
    synstruta: float
    synstrutt: float
    drnp: float
    drvp: float
    drap: float
    drpp: float
    drpval: float
    drneg: float
    drgerund: float
    drinf: float
    wrdnoun: float
    wrdverb: float
    wrdadj: float
    wrdadv: float
    wrdpro: float
    wrdprp1s: float
    wrdprp1p: float
    wrdprp2: float
    wrdprp2s: float
    wrdprp2p: float
    wrdprp3s: float
    wrdprp3p: float
    wrdfrqc: float
    wrdfrqa: float
    wrdfrqmc: float
    wrdaoac: float
    wrdfamc: float
    wrdcncc: float
    wrdimgc: float
    wrdmeac: float
    rdfre: float
    rdfkgl: float
    rdl2: float


class CohMetrixExtractor(FeatureExtractor):
    def extract(self, text: str, **kwargs) -> CohMetrixFeatures:
        """Esse método realiza a extração das características
        do CohMetrix BR para o texto recebido como argumento.
        """
        del kwargs

        return CohMetrixFeatures(**{f.__name__.lower(): float(f(text))
                                    for f in cohmetrixBR.FEATURES})
