import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Configuração da página
st.set_page_config(
    page_title="Predição de Empregabilidade | IA",
    page_icon="💼",
    layout="centered"
)

# Carregar o modelo treinado (pipeline com pré-processamento incluso)
@st.cache_resource
def carregar_modelo():
    caminhos = [
        os.path.join("modelo", "modelo_emprego.pkl"),
        os.path.join("modelo", "modelo_emprego.sav"),
        os.path.join("modelo", "logistic_model.pkl"),
        "modelo_emprego.pkl",
        "modelo_emprego.sav"
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            with open(caminho, "rb") as model_file:
                modelo = pickle.load(model_file)
            return modelo, caminho
    return None, None

modelo, caminho_encontrado = carregar_modelo()

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

if modelo is None:
    st.warning("⚠️ **Arquivo do modelo não encontrado na pasta `modelo/`!**")
    st.info("""
    Para utilizar a predição online:
    1. Baixe o arquivo **`modelo_emprego.sav`** gerado no Google Colab (`/content/drive/MyDrive/Lab_IA/Dataset/modelo_emprego.sav`).
    2. Coloque-o dentro da pasta **`modelo/`** deste projeto com o nome `modelo_emprego.pkl` (ou `.sav`).
    3. Ou execute localmente `python treinamento.py` com o dataset na pasta `dataset/`.
    """)
    st.stop()

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
    if st.button("🚀 Prever Situação Profissional", type="primary", use_container_width=True):
        # Montar DataFrame de entrada no formato esperado pelo pipeline scikit-learn
        novo_perfil = pd.DataFrame([{
            "Age": age_selecionada,
            "Education": education_selecionada,
            "YearsCode": years_code
        }])

        try:
            # Predição da classe
            previsao = modelo.predict(novo_perfil)[0]

            # Obtenção das probabilidades
            probabilidades = modelo.predict_proba(novo_perfil)[0]
            
            # Identificação do índice da classe 1 (Empregado)
            if hasattr(modelo, "named_steps") and "modelo" in modelo.named_steps:
                classes_modelo = modelo.named_steps["modelo"].classes_
            elif hasattr(modelo, "classes_"):
                classes_modelo = modelo.classes_
            else:
                classes_modelo = [0, 1]

            indice_empregado = list(classes_modelo).index(1)
            prob_empregado = probabilidades[indice_empregado]
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

        except Exception as e:
            st.error(f"Erro ao processar a predição: {e}")

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
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

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
