# -*- coding: utf-8 -*-
"""
Treinamento do Modelo de Predição de Empregabilidade
Disciplina: Tópicos Avançados em Inteligência Artificial - IFSP Campus Catanduva
Dataset: Stack Overflow Developer Survey
Modelo: Regressão Logística Balanceada
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

SEED = 42

def encontrar_dataset():
    """Localiza o dataset em caminhos prováveis."""
    candidatos = [
        os.path.join("dataset", "survey_results_public.csv"),
        "survey_results_public.csv",
        os.path.join("..", "dataset", "survey_results_public.csv"),
        os.path.expanduser(r"~\Downloads\survey_results_public.csv"),
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho
    return None

def main():
    print("=" * 60)
    print("Iniciando treinamento do modelo de Empregabilidade")
    print("=" * 60)

    caminho_csv = encontrar_dataset()
    if not caminho_csv:
        raise FileNotFoundError(
            "Arquivo 'survey_results_public.csv' não foi encontrado.\n"
            "Por favor, coloque o dataset dentro da pasta 'dataset/'."
        )

    print(f"Lendo dataset de: {caminho_csv}")
    colunas_necessarias = ["Age", "Education", "YearsCode", "Employment"]
    
    # Leitura otimizada carregando apenas as colunas necessárias
    dataset = pd.read_csv(caminho_csv, usecols=colunas_necessarias, low_memory=False)
    print(f"Total de registros lidos: {len(dataset):,}")

    # Limpeza dos textos
    for coluna in ["Age", "Education", "Employment"]:
        dataset[coluna] = dataset[coluna].apply(
            lambda v: v.strip() if isinstance(v, str) else v
        )

    # Conversão de YearsCode para numérico
    dataset["YearsCode"] = pd.to_numeric(dataset["YearsCode"], errors="coerce")

    # Definição das classes de empregabilidade
    empregados = [
        "Employed",
        "Independent contractor, freelancer, or self-employed"
    ]
    nao_empregados = [
        "Student",
        "Not employed"
    ]

    # Filtrar apenas respondentes dentro dessas categorias
    dados = dataset[dataset["Employment"].isin(empregados + nao_empregados)].copy()

    # Criação da variável-alvo: 1 = Empregado, 0 = Não empregado
    dados["Empregado"] = dados["Employment"].isin(empregados).astype(int)

    print("\nDistribuição da variável-alvo:")
    print(dados["Empregado"].value_counts())
    print((dados["Empregado"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

    X = dados[["Age", "Education", "YearsCode"]]
    y = dados["Empregado"]

    # Divisão treino/teste (80/20) estratificada
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    print(f"\nAmostras de Treino: {len(X_treino):,} | Teste: {len(X_teste):,}")

    # Pipeline de Pré-processamento
    preprocessamento_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler())
    ])

    preprocessamento_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessamento = ColumnTransformer([
        ("numerico", preprocessamento_numerico, ["YearsCode"]),
        ("categorico", preprocessamento_categorico, ["Age", "Education"])
    ])

    # Classificador: Regressão Logística com pesos balanceados
    classifier = Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", LogisticRegression(
            random_state=SEED,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("\nTreinando o modelo de Regressão Logística Balanceada...")
    classifier.fit(X_treino, y_treino)
    print("Modelo treinado com sucesso!")

    # Avaliação no conjunto de teste
    y_pred = classifier.predict(X_teste)
    acc = accuracy_score(y_teste, y_pred)
    acc_bal = balanced_accuracy_score(y_teste, y_pred)
    prec = precision_score(y_teste, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_teste, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_teste, y_pred, pos_label=1, zero_division=0)

    print("\n" + "=" * 40)
    print("MÉTRICAS NO CONJUNTO DE TESTE:")
    print(f"Acurácia Geral:       {acc:.2%}")
    print(f"Acurácia Balanceada:  {acc_bal:.2%}")
    print(f"Precisão (Empregado): {prec:.2%}")
    print(f"Recall (Empregado):   {rec:.2%}")
    print(f"F1-Score (Empregado): {f1:.2%}")
    print("=" * 40)

    # Criar pasta modelo/ caso não exista
    os.makedirs("modelo", exist_ok=True)
    caminho_saida = os.path.join("modelo", "modelo_emprego.pkl")

    with open(caminho_saida, "wb") as f:
        pickle.dump(classifier, f)

    print(f"\nModelo salvo com sucesso em: {caminho_saida}")

if __name__ == "__main__":
    main()
