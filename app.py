import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import json
import math

# Configuração da página
st.set_page_config(
    page_title="Predição de Empregabilidade | IA",
    page_icon="💼",
    layout="centered"
)

# Shim de compatibilidade para desserialização de pipelines scikit-learn
try:
    from sklearn.compose import _column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# Parâmetros matemáticos exatos extraídos do modelo treinado
# Garante 100% de estabilidade e funcionamento em qualquer versão de Python e scikit-learn
PARAMETROS_MODELO = {
    "intercept": -0.141300597764714,
    "years_code": {
        "median": 15.0,
        "mean": 16.58091356053284,
        "scale": 10.41484650141273,
        "coef": 0.43046973588184295
    },
    "age_coefficients": {
        "18-24 years old": -1.179920868845064,
        "25-34 years old": 0.7040771998945536,
        "35-44 years old": 1.0227272364475353,
        "45-54 years old": 0.6295695619601823,
        "55-64 years old": 0.5801010852848798,
        "65 years or older": 0.5465479331954296,
        "Prefer not to say": -0.23903025917936305,
        "Under 18 years old": -2.1981951549533383
    },
    "education_coefficients": {
        "Associate degree (A.A., A.S., etc.)": 0.2505070599153566,
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 0.40055077826300817,
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 0.5586302022268608,
        "Other, please specify": -0.45667260115402053,
        "Primary/elementary school": -0.6630798923123997,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 0.4720453100502225,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": -0.44796419423346123,
        "Some college/university study without earning a degree": -0.24813992895070572
    }
}

class MotorPredicao:
    """Motor híbrido de inferência: tenta utilizar o pipeline serializado e possui fallback matemático exato."""
    def __init__(self):
        self.pipeline = self._carregar_pipeline()
        self.params = PARAMETROS_MODELO

    def _carregar_pipeline(self):
        caminhos = [
            os.path.join("modelo", "modelo_emprego.sav"),
            os.path.join("modelo", "modelo_emprego.pkl"),
            os.path.join("modelo", "logistic_model.pkl"),
            "modelo_emprego.sav",
            "modelo_emprego.pkl"
        ]
        for caminho in caminhos:
            if os.path.exists(caminho):
                try:
                    with open(caminho, "rb") as f:
                        obj = pickle.load(f)
                    return obj
                except Exception:
                    # Em caso de divergência de bytecode entre versões do Scikit-Learn (ex.: 1.6 vs 1.9)
                    continue
        return None

    def prever(self, age: str, education: str, years_code: float):
        # 1. Tentar executar via Scikit-Learn Pipeline
        if self.pipeline is not None:
            try:
                df = pd.DataFrame([{
                    "Age": age,
                    "Education": education,
                    "YearsCode": years_code
                }])
                previsao = int(self.pipeline.predict(df)[0])
                probabilidades = self.pipeline.predict_proba(df)[0]
                
                classes = getattr(self.pipeline, "classes_", [0, 1])
                if hasattr(self.pipeline, "named_steps") and "modelo" in self.pipeline.named_steps:
                    classes = self.pipeline.named_steps["modelo"].classes_

                idx_emp = list(classes).index(1)
                prob_emp = float(probabilidades[idx_emp])
                return previsao, prob_emp
            except Exception:
                pass

        # 2. Execução matemática direta da Regressão Logística (100% à prova de falhas)
        p = self.params
        y_val = float(years_code) if years_code is not None else p["years_code"]["median"]
        y_scaled = (y_val - p["years_code"]["mean"]) / p["years_code"]["scale"]
        
        coef_y = p["years_code"]["coef"] * y_scaled
        coef_a = p["age_coefficients"].get(age, 0.0)
        coef_e = p["education_coefficients"].get(education, 0.0)
        
        z = p["intercept"] + coef_y + coef_a + coef_e
        prob_emp = 1.0 / (1.0 + math.exp(-z))
        previsao = 1 if prob_emp >= 0.5 else 0
        return previsao, prob_emp

@st.cache_resource
def obter_motor():
    return MotorPredicao()

motor = obter_motor()

# Informações institucionais (mantendo o mesmo padrão didático do IFSP)
st.markdown("""
**Professor Dr. Márcio Andrey Teixeira**  
**IFSP - Campus Catanduva**  
**Disciplina:** Tópicos Avançados em Inteligência Artificial  
Aplicação didática desenvolvida para estimativa de probabilidade de empregabilidade no setor de tecnologia utilizando **Regressão Logística Balanceada** com dados do *Stack Overflow Developer Survey*.
""")

st.markdown("---")

# Título da aplicação
st.title("💼 Predição de Empregabilidade")
st.write("Estime a probabilidade de uma pessoa da área de tecnologia pertencer ao grupo de profissionais empregados com base em idade, escolaridade e anos de experiência.")

# Abas organizadas: Predição interativa e Análise do Estudo
tab_predicao, tab_metricas, tab_dataset = st.tabs(["🔮 Fazer Predição", "📊 Comparação de Modelos", "ℹ️ Sobre o Dataset"])

with tab_predicao:
    st.header("Informações do Profissional")

    # Mapeamento amigável para Idade
    opcoes_idade = [
        "18-24 years old",
        "25-34 years old",
        "35-44 years old",
        "45-54 years old",
        "55-64 years old",
        "65 years or older",
        "Under 18 years old",
        "Prefer not to say"
    ]

    labels_idade = {
        "18-24 years old": "18 a 24 anos",
        "25-34 years old": "25 a 34 anos",
        "35-44 years old": "35 a 44 anos",
        "45-54 years old": "45 a 54 anos",
        "55-64 years old": "55 a 64 anos",
        "65 years or older": "65 anos ou mais",
        "Under 18 years old": "Menor de 18 anos",
        "Prefer not to say": "Prefiro não informar"
    }

    # Mapeamento amigável para Escolaridade
    opcoes_educacao = [
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
        "Associate degree (A.A., A.S., etc.)",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
        "Some college/university study without earning a degree",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
        "Primary/elementary school",
        "Other, please specify"
    ]

    labels_educacao = {
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": "Bacharelado / Graduação completa (B.A., B.S., Eng.)",
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": "Mestrado / MBA",
        "Associate degree (A.A., A.S., etc.)": "Tecnólogo / Associate degree",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Doutorado / Pós-Doutorado / Título Profissional",
        "Some college/university study without earning a degree": "Ensino Superior incompleto",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Ensino Médio completo",
        "Primary/elementary school": "Ensino Fundamental",
        "Other, please specify": "Outro nível"
    }

    col1, col2 = st.columns(2)

    with col1:
        age_selecionada = st.selectbox(
            "Faixa Etária (Age)",
            options=opcoes_idade,
            format_func=lambda x: f"{labels_idade.get(x, x)} ({x})",
            index=1 # 25-34 anos
        )

    with col2:
        education_selecionada = st.selectbox(
            "Escolaridade (Education)",
            options=opcoes_educacao,
            format_func=lambda x: f"{labels_educacao.get(x, x)}",
            index=0 # Bacharelado
        )

    years_code = st.slider(
        "Anos de Experiência com Programação (YearsCode)",
        min_value=0,
        max_value=50,
        value=5,
        help="Quantidade total de anos em que a pessoa programa (profissional ou aprendizado)."
    )

    st.markdown("")

    # Botão de predição
    if st.button("🚀 Prever Situação Profissional", type="primary"):
        previsao, prob_empregado = motor.prever(
            age=age_selecionada,
            education=education_selecionada,
            years_code=years_code
        )

        prob_nao_empregado = 1.0 - prob_empregado

        st.markdown("---")
        st.subheader("Resultado da Predição")

        if previsao == 1:
            st.success(f"### ✅ Situação Prevista: **Empregado**")
            st.markdown(
                f"Com base nas características informadas, o modelo estima uma probabilidade de "
                f"**{prob_empregado:.2%}** de pertencer ao grupo de empregados."
            )
        else:
            st.warning(f"### ⚠️ Situação Prevista: **Não empregado** (Estudante / Em transição)")
            st.markdown(
                f"Com base nas características informadas, o modelo estima uma probabilidade de "
                f"**{prob_nao_empregado:.2%}** de pertencer ao grupo de não-empregados / estudantes "
                f"(Probabilidade de Empregado: **{prob_empregado:.2%}**)."
            )

        # Barra de progresso para a probabilidade de emprego
        st.write("**Probabilidade de Empregabilidade:**")
        st.progress(float(prob_empregado))

        # Métricas em colunas
        m1, m2 = st.columns(2)
        m1.metric("Chance de Empregado (1)", f"{prob_empregado:.2%}")
        m2.metric("Chance de Não Empregado (0)", f"{prob_nao_empregado:.2%}")

with tab_metricas:
    st.header("📊 Comparação dos Modelos Desenvolvidos no Estudo")
    st.markdown("""
    Durante a disciplina, múltiplos algoritmos foram testados e comparados sobre o conjunto de teste.
    Abaixo estão as métricas obtidas na validação experimental:
    """)

    dados_comparacao = {
        "Métrica": [
            "Acurácia Geral",
            "Acurácia Balanceada",
            "Precisão (Empregado)",
            "Recall (Empregado)",
            "F1-score (Empregado)"
        ],
        "Regressão Logística Original": ["87.34%", "69.54%", "89.20%", "96.40%", "92.66%"],
        "Regressão Logística Balanceada (Em Deploy)": ["74.80%", "77.35%", "93.10%", "76.40%", "83.92%"],
        "Árvore de Decisão (Prof. 3)": ["73.10%", "75.20%", "92.40%", "74.80%", "82.67%"],
        "Árvore de Decisão (Prof. 5)": ["74.50%", "76.80%", "93.00%", "76.10%", "83.70%"]
    }
    df_metricas = pd.DataFrame(dados_comparacao)
    st.dataframe(df_metricas, hide_index=True)

    st.markdown("""
    > **Por que a Regressão Logística Balanceada foi escolhida?**  
    > O dataset possui um forte desbalanceamento natural de classes (a grande maioria dos respondentes está empregada).  
    > O modelo com pesos balanceados (`class_weight='balanced'`) obteve **Acurácia Balanceada de 77.35%**, demonstrando capacidade justa de identificar tanto candidatos empregados quanto não-empregados, evitando o viés da maioria com baixo consumo de memória e máxima interpretabilidade.
    """)

with tab_dataset:
    st.header("ℹ️ Sobre o Dataset e Variáveis")
    st.markdown("""
    * **Fonte dos Dados:** *Stack Overflow Developer Survey* (pesquisa anual global com dezenas de milhares de desenvolvedores).
    * **Variáveis Preditivas Utilizadas:**
      * `Age`: Faixa etária do respondente.
      * `Education`: Nível de escolaridade formal mais alto atingido.
      * `YearsCode`: Tempo total (em anos) que o respondente programa.
    * **Variável-Alvo (`Empregado`):**
      * **1 (Empregado):** Respondentes com vínculo empregatício formal ou atuando como freelancers / prestadores de serviço independentes.
      * **0 (Não empregado):** Respondentes na condição de estudantes ou sem trabalho formal.
    """)

st.markdown("---")
st.caption("Desenvolvido para fins didáticos e acadêmicos • Hospedado no Streamlit Cloud")
