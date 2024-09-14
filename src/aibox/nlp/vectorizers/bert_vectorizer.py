"""Esse módulo contém um vetorizador
baseado no BERT.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, models

from aibox.nlp.core import Vectorizer


class BertVectorizer(Vectorizer):
    def __init__(
        self,
        bert_model: SentenceTransformer = None,
        sentence_bert_name: str = None,
        pooling_type: str = "cls",
        max_seq_len: int = 512,
        do_lower_case: bool = False,
        tokenizer_name: str = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> None:
        if bert_model is None:
            if sentence_bert_name is None:
                sentence_bert_name = "neuralmind/bert-base-portuguese-cased"

            word_embedding_model = models.Transformer(
                sentence_bert_name,
                max_seq_length=max_seq_len,
                do_lower_case=do_lower_case,
                tokenizer_name_or_path=tokenizer_name,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), pooling_type
            )
            bert_model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], device=device
            )
        self.model = bert_model
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def _vectorize(self, text: str):
        return self.model.encode(
            sentences=text,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )
