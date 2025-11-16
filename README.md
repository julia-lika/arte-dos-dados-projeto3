# Documentação do Pipeline de Machine Learning

Este documento detalha as etapas de pré-processamento, Análise Exploratória de Dados (EDA) e modelagem aplicadas ao *dataset* de sinistros de seguro de carro (*Car Insurance Claim*).

-----

## 1\. Limpeza e Padronização dos Nomes das Colunas

O objetivo é garantir que os nomes das colunas sigam um padrão consistente para facilitar a manipulação e evitar erros.

### Implementação: Função `clean_column_names` (Célula 1)

Essa função auxiliar realiza duas ações principais:

  * **Aplicar lowercase em todas as colunas:** Todos os nomes de colunas são convertidos para letras minúsculas.
  * **Excluir caracteres especiais dos nomes das colunas:** Espaços (`     `) são substituídos por *underscores* (`_`), e a função remove quaisquer caracteres que não sejam alfanuméricos ou *underscores*.

```python
# Trecho da implementação
def clean_column_names(df):
    df = df.copy()
    # Aplica lowercase e substitui espaços por underscore
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Remove caracteres não alfanuméricos (mantém underscores)
    df.columns = [''.join(ch for ch in c if ch.isalnum() or ch == '_') for c in df.columns]
    return df

df = clean_column_names(df)
```

-----

## 2\. Tratamento de Outliers (Célula 3)

Outliers nas variáveis numéricas foram tratados utilizando a técnica de **Winsorização Interquartil (IQR Winsorization)** para limitar valores extremos sem eliminá-los, preservando o tamanho da amostra.

### Implementação: Função `winsorize_iqr`

1.  **Cálculo dos Limites:** Para cada coluna numérica, são calculados o primeiro quartil ($Q_1$), o terceiro quartil ($Q_3$) e o Intervalo Interquartil ($IQR = Q_3 - Q_1$).
2.  **Limitação:** Os limites de corte são definidos como $Q_1 - 1.5 \times IQR$ (limite inferior) e $Q_3 + 1.5 \times IQR$ (limite superior).
3.  **Winsorização:** Todos os valores abaixo do limite inferior são substituídos pelo limite inferior, e todos os valores acima do limite superior são substituídos pelo limite superior, usando o método `.clip()`.


```python
# Trecho da implementação (Célula 3)
def winsorize_iqr(df, cols):
    # ... (cálculo de Q1, Q3, IQR, lower, upper)
    df[col] = df[col].clip(lower, upper)
    # ...
```

-----

## 3\. Pré-processamento e Engenharia de Features (Célula 6)

O tratamento de *missing values* e a codificação de dados categóricos foram realizados dentro de pipelines da biblioteca `scikit-learn` para evitar *data leakage* (vazamento de dados) e garantir que a lógica seja aplicada consistentemente aos conjuntos de treino e teste/submissão.

### 3.1. Tratamento dos Missing Values

A imputação foi definida separadamente para colunas numéricas e categóricas:

  * **Colunas Numéricas:** Imputação com a **mediana** (`SimpleImputer(strategy='median')`). A mediana é menos sensível a outliers (já tratados).
  * **Colunas Categóricas:** Imputação com a **moda** (`SimpleImputer(strategy='most_frequent')`).

### 3.2. Lidar com Dados Categóricos

  * As colunas categóricas são processadas usando o **OneHotEncoder** (Codificação One-Hot) para transformá-las em formato numérico, criando uma nova coluna binária para cada categoria única.

### 3.3. Coluna `ID` e Estrutura

  * A coluna **`id`** é excluída dos conjuntos de treino e teste (`X_train`, `X_test`) na **Célula 5**, mas é salva para uso no arquivo de submissão na **Célula 10**.
  * A pré-processamento é orquestrado pelo **`ColumnTransformer`**, combinando os pipelines numérico (Imputer + StandardScaler) e categórico (Imputer + OneHotEncoder).

-----

## 4\. Análise Exploratória de Dados (EDA) (Célula 4)

A EDA foi uma etapa expandida para entender a distribuição dos dados, a relação entre *features* e a variável *target*, e identificar padrões.

  * **Target Distribution:** Visualização da proporção da variável **`outcome`** (sinistro) para verificar o nível de desbalanceamento.
  * **Distribuições Numéricas:** Uso de **histogramas** para visualizar a distribuição de variáveis como *age* e *credit\_score*.
  * **Taxa de Risco Categórica:** Gráficos de **barras** para calcular e visualizar a taxa média de sinistro para cada categoria (ex: *gender* vs. *outcome*).
  * **Heatmap de Correlação:** Geração de um **mapa de calor** para visualizar a correlação entre as *features* numéricas e a variável *target*.
  * **Boxplots:** **Boxplots** comparando as distribuições de *features* numéricas (*age*, *annual\_premium*, etc.) em relação à classe *target* (0 ou 1).
  * **Contagem de Categorias:** Gráficos de barras para visualizar a frequência de todas as categorias nas colunas categóricas.

-----

## 5\. Modelagem e Seleção (Células 5, 7-10)

### 5.1. Definição das Amostras (Célula 5)

  * O *dataset* foi dividido em conjuntos de treino e teste (`X_train`, `X_test`, `y_train`, `y_test`) usando **`train_test_split`** com proporção **80/20**.
  * Foi usado o parâmetro **`stratify=y`** para garantir que a proporção da classe *target* (balanceamento/desbalanceamento) seja mantida igual nos conjuntos de treino e teste.

### 5.2. Seleção de Features (Célula 7)

  * **Método:** Foi utilizado **`SelectKBest`** em conjunto com a métrica **`mutual_info_classif`** (Informação Mútua para Classificação).
  * **Objetivo:** Selecionar as **30 melhores *features*** que possuem a maior dependência estatística (Informação Mútua) com a variável *target*. Essa etapa é crucial para reduzir a dimensionalidade e o ruído.

### 5.3. Algoritmos e Métricas (Células 8 e 9)

  * **Algoritmos Escolhidos:**
      * **`LogisticRegression`** (Regressão Logística): Modelo linear simples e interpretável.
      * **`RandomForestClassifier`** (Random Forest): Modelo de *ensemble* não-linear robusto, eficaz contra *overfitting*.
  * **Métricas de Performance:** A função **`evaluate`** (Célula 8) calcula:
      * **Acurácia**
      * **Precisão**
      * **Recall**
      * **F1-Score** (Métrica principal para classes desbalanceadas)
      * **ROC AUC** (Área sob a curva ROC)
      * **Matriz de Confusão**

### 5.4. Seleção do Melhor Modelo Preditivo (Célula 10)

  * **Critério:** O modelo com o **maior F1-Score** no conjunto de teste é selecionado como o melhor modelo preditivo. O F1-Score é a média harmônica de Precision e Recall e é o critério mais adequado para otimizar modelos em *datasets* com classes desbalanceadas (como é comum em problemas de sinistro/fraude).
  * **Output:** O pipeline completo do modelo vencedor é salvo no disco usando `joblib.dump()`.

-----

## 6\. Geração do Arquivo de Submissão (Célula 10)

A etapa final utiliza o `best_pipeline` treinado e selecionado para gerar as predições no arquivo `sample_submission.csv`.

1.  O arquivo de submissão (`sample_submission.csv`) é carregado e tem suas colunas limpas.
2.  A coluna **`id`** (`policy_id`) é separada.
3.  As *features* do arquivo de submissão (`test_features_no_id`) são passadas para o `best_pipeline`.
4.  O pipeline aplica o pré-processamento (imputação, escalonamento, one-hot encoding) e a seleção das 30 *features* automaticamente, garantindo que o conjunto de teste seja transformado exatamente como o conjunto de treino.
5.  O modelo faz a predição (`y_pred_submission`).
6.  As predições e os IDs são combinados em um DataFrame final e salvos em `submission_[modelo].csv`.