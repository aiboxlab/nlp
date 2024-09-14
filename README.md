<h1 align="center">
  <br>
  <a href="https://aiboxlab.org/en/"><img src="https://aiboxlab.org/img/logo-aibox.png" alt="AiBox Lab" width="200"></a>
  <br>
  aibox-nlp
  <br>
</h1>

<h4 align="center">Uma biblioteca de Processamento de Linguagem Natural para o Português Brasileiro.</h4>

<p align="center">
  <a href="#funcionalidades">Funcionalidades</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#instalação">Instalação</a>
</p>


## Funcionalidades

* **315+ características textuais** para o Português Brasileiro;
  * CohMetrix-BR, NILCMetrix, Características Gramaticais, e outras!
* **Classificadores e Regressores** clássicos
  * SVM, SVR, XGBoost, CatBoost, LGBM, RF, e outros!
* Classificação e Regressão com **Deep Learning**
  * BERT, LSTM, BI-LSTM, CharCNN, entre outros!
* **Várias Estratégias de Vetorização**
  * Vetorização baseada em Embeddings (nível de sentença, nível de palavra), baseada em TF-IDF, e outros!
* **Reprodutibilidade**
  * Todos experimentos são reprodutíveis, basta indica a seed
* **AutoML**: experimentação automática, basta indicar o conjunto de dados
  * Backend com `optuna` para otimização de parâmetros e motores de busca

## Quick Start

A biblioteca se baseia em 3 entidades básicas:

* **Dataset**
  * Um dataset representa um conjunto de pares de **textos** e **targets** (classes, ou valores), que devem ser utilizados para resolver um problema de classificação ou regressão.
* **Metric**
  * Uma métrica permite as saídas de um dado estimador com os valores ground-truth do dataset.
  * Por exemplo, Precisão, Revocação e F1-score são métricas para avaliação.
  * Também existem outras métricas como o Kappa e Kappa Vizinho.
* **Pipeline**
  * Representam um conjunto de 3 componentes:
    1. **Estratégia de Vetorização**
       * Converte um texto para sua representação numérica.
       * Alguns exemplos são extratores de características, extração de Embeddings (BERT, FastText, etc), ou TF-IDF.
    2. **Estimador**
       * Representam um algoritmo para classificação/regressão.
       * Alguns exemplos são SVM, SVR, Árvores de Decisão, Redes Neurais.
    3. **Pós-processamento**
       * Estratégia aplicada após a predição pelo estimador.
       * Pode ser utilizada para garantir os limites da saída, ou conversão de regressão para classificação.

Um **Experimento** permite comparar múltiplas **Pipelines** com as **Métricas** escolhidas em um dado **Dataset**. Para construir um experimento, é possível utilizar as classes presentes em `aibox.nlp.experiments` ou utilizar os padrões factory/builder presentes em `aibox.nlp.factory`. Um exemplo básico pode ser encontrado abaixo:

```python
from aibox.nlp.factory.experiment import SimpleExperimentBuilder

# === Construindo um experimento para classificação no Essay-BR ===
# Por simplicidade, vamos instanciar um experimento
#   para comparar algumas abordagens para classificação
#   da competência 1 do dataset Essay-BR.
builder = SimpleExperimentBuilder()

# Inicialmente, vamos definir o dataset
builder.dataset('essayBR',
                extended=False,
                target_competence='C1')

# Vamos definir o tipo do problema
builder.classification()

# Vamos definir a seed randômica
builder.seed(42)

# Depois, vamos definir algumas métricas
#   que devem ser calculadas
builder.add_metric('precision', average='weighted')
builder.add_metric('recall', average='weighted')
builder.add_metric('f1', average='weighted')
builder.add_metric('kappa')
builder.add_metric('neighborKappa')

# Depois, vamos definir qual a métrica
#   que deve ser utilizar para escolher a
#   melhor pipeline
builder.best_criteria('precision', average='weighted')

# Agora, vamos adicionar algumas pipelines baseadas
#   em extração de característica
builder.add_feature_pipeline(
    features=['textualSimplicityBR'],
    estimators=['svm'],
    names=['svm+textual_simplicity'])

builder.add_feature_pipeline(
    features=['readabilityBR'],
    estimators=['svm'],
    names=['svm+readability'])

# Uma vez que tenhamos configurado o experimento,
#   podemos obter uma instância:
experiment = builder.build()

# === Executando o experimento ===
result = experiment.run()

# === Inspecionando os resultados ===
result.best_pipeline_name
# svm+textual_simplicity

result.best_metrics_history
# {
#   "svm+textual_simplicity": {
#     "Weighted Precision": 0.33119142,
#     "Weighted Recall": 0.5754923,
#     "Weighted F1-score": 0.42042914,
#     "Kappa": 0.0,
#     "Neighbor Kappa": 0.0
#   },
#   "svm+readability": {
#     "Weighted Precision": 0.33119142,
#     "Weighted Recall": 0.5754923,
#     "Weighted F1-score": 0.42042914,
#     "Kappa": 0.0,
#     "Neighbor Kappa": 0.0
#   }
# }
```

Para mais exemplos, acesse a [documentação](examples).


## Instalação

Primeiro, realiza a instalação da biblioteca via `pip` ou através do `git clone`:

### 1. Instalando com o pip

```bash
# Configurar ambiente virtual
# ...

# Instalar através do pip
$ pip install aibox-nlp

# Adicionalmente, instalar dependências opcionais:

# BR contém características para PT-BR
$ pip install aibox-nlp[BR]

# trees contém estimadores baseados em árvore
$ pip install aibox-nlp[trees]

# embeddings contém vetorizadores baseados em modelos
$ pip install aibox-nlp[embeddings]

# Ou, instalar todas:
$ pip install aibox-nlp[all]
```

### 2. Instalando localmente

```bash
# Clonar repositório
$ git clone https://github.com/aiboxlab/nlp

# Acessar diretório
$ cd nlp

# Configurar ambiente virtual
# ...

# Instalar através do pip
$ pip install -e .

# Adicionalmente, instalar dependências
#   desejadas opcionais
$ pip install .[BR]
$ pip install .[trees]
$ pip install .[embeddings]

# Também é possível baixar todas as opcionais:
$ pip install .[all]
```

## License

MIT

---
