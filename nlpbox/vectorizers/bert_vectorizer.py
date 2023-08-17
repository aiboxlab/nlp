from typing import Optional
from nlpbox.core import Vectorizer
from sentence_transformers import SentenceTransformer, models

class BertVectorizer(Vectorizer):
    def __init__(self,bert_model:Optional[SentenceTransformer] = None,
                  sentence_bert_name:Optional[str] = 'neuralmind/bert-base-portuguese-cased',
                  pooling_type:str ='cls',
                  max_seq_len:int = 512,
                  do_lower_case:bool = False,
                  tokenizer_name_or_path:Optional[str] = None,
                  batch_size:int = 32,
                  show_progress_bar:Optional[bool] = False,
                  device: Optional[str] = None, 
                  normalize_embeddings: bool = False
                  ) -> None:
        super().__init__()
        if bert_model is None:
            if sentence_bert_name is None:
                raise ValueError('No model or name was passed')
            word_embedding_model = models.Transformer(sentence_bert_name, max_seq_length=max_seq_len,do_lower_case=do_lower_case,tokenizer_name_or_path=tokenizer_name_or_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling_type)
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device)
        elif isinstance(bert_model,SentenceTransformer):
            self.model = bert_model
        else:
            raise TypeError('Model was not instance of SentenceTransformer object')
        
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        
    def _vectorize(self,text):
            return self.model.encode(sentences=text,
                              batch_size=self.batch_size,
                              show_progress_bar=self.show_progress_bar,
                              convert_to_numpy=True,
                              device=self.device,
                              normalize_embeddings=self.normalize_embeddings
                              )
