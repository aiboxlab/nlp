<h1 align="center">
  <br>
  <a href="https://aiboxlab.org/en/"><img src="https://aiboxlab.org/img/logo-aibox.png" alt="AiBox Lab" width="200"></a>
  <br>
  nlpbox
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

```python
from nlpbox.factory.experiment import SimpleExperimentBuilder

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
```

## Instalação

Primeiro, realiza a instalação da biblioteca via `pip` ou através do `git clone`:

```bash
# Configura ambiente virtual
# ...

# Instalar através do pip
$ pip install aibox-nlpbox

# Adicionalmente, instalar dependências opcionais
$ pip install aibox-nlpbox[BR]
$ pip install aibox-nlpbox[trees]
$ pip install aibox-nlpbox[embeddings]
```

```bash
# Clonar repositório
$ git clone https://github.com/aiboxlab/nlpbox

# Acessar diretório
$ cd nlpbox

# Configurar ambiente virtual
# ...

# Instalar através do pip
$ pip install -e .

# Adicionalmente, instalar dependências opcionais
$ pip install .[BR]
$ pip install .[trees]
$ pip install .[embeddings]
```

## License

MIT

---
