# 💼 Predição de Empregabilidade com Inteligência Artificial

Aplicação web interativa desenvolvida em **Streamlit** e **Scikit-Learn** para estimar a probabilidade de um profissional do setor de tecnologia pertencer ao grupo de empregados, com base em sua faixa etária, escolaridade e anos de experiência em programação.

Projeto acadêmico desenvolvido para a disciplina de **Tópicos Avançados em Inteligência Artificial** no **IFSP - Campus Catanduva**, sob orientação do **Prof. Dr. Márcio Andrey Teixeira**.

---

## 📁 Estrutura do Projeto

```text
app_Predicao_Empregabilidade/
│
├── dataset/
│   └── README.md              # Explicação sobre o dataset e limites do GitHub
│
├── modelo/
│   ├── README.md              # Instruções sobre o artefato do modelo
│   └── modelo_emprego.pkl     # Pipeline serializado (Pré-processador + Regressão Logística)
│
├── .gitignore                 # Arquivos ignorados pelo Git (ignora CSVs >100MB)
├── app.py                     # Aplicação web Streamlit
├── requirements.txt           # Dependências para deploy no Streamlit Cloud
├── treinamento.py             # Script de preparação e treinamento do modelo
└── README.md                  # Documentação do projeto
```

---

## 🧠 Metodologia e Escolha do Modelo

O conjunto de dados utilizado é o **Stack Overflow Developer Survey**.  
Variáveis utilizadas:
* `Age` (Faixa etária)
* `Education` (Nível educacional)
* `YearsCode` (Tempo total em anos de experiência com programação)

A variável-alvo `Empregado` foi binarizada:
* **1 (Empregado):** Profissionais com emprego em tempo integral/parcial ou atuando de forma independente/freelancer.
* **0 (Não empregado):** Estudantes ou pessoas sem ocupação no momento da pesquisa.

### Comparativo dos Modelos Experimentados:

| Modelo | Acurácia Geral | Acurácia Balanceada | F1-Score (Empregado) |
|---|:---:|:---:|:---:|
| Regressão Logística Padrão | 87.34% | 69.54% | 92.66% |
| **Regressão Logística Balanceada (Deploy)** | **74.80%** | **77.35%** | **83.92%** |
| Árvore de Decisão (Profundidade 3) | 73.10% | 75.20% | 82.67% |
| Árvore de Decisão (Profundidade 5) | 74.50% | 76.80% | 83.70% |

> **Justificativa da Escolha:** A **Regressão Logística Balanceada** (`class_weight='balanced'`) obteve o melhor equilíbrio na **Acurácia Balanceada (77.35%)**, tratando o desbalanceamento inerente do mercado de trabalho de forma justa sem ignorar a classe minoritária, oferecendo interpretabilidade transparente e consumo de recursos mínimo no servidor de deploy.

---

## 🚀 Como Executar Localmente

### 1. Clonar ou navegar até a pasta do projeto:
```bash
cd app_Predicao_Empregabilidade
```

### 2. Criar e ativar um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

### 3. Instalar as dependências:
```bash
pip install -r requirements.txt
```

### 4. Executar a aplicação Streamlit:
```bash
streamlit run app.py
```

---

## 🌐 Como Hospedar Gratuitamente no Streamlit Cloud via GitHub

### Passo 1: Subir o projeto para o GitHub
1. Crie um novo repositório no seu GitHub (exemplo: `app-predicao-empregabilidade`), deixe-o como **Public**.
2. Na pasta do projeto em seu terminal (ou via **GitHub Desktop**):
   ```bash
   git init
   git add .
   git commit -m "Deploy: app de predição de empregabilidade"
   git branch -M main
   git remote add origin https://github.com/SEU_USUARIO/app-predicao-empregabilidade.git
   git push -u origin main
   ```
   *(Certifique-se de que o arquivo `modelo/modelo_emprego.pkl` esteja comitado e que o CSV gigante de 122MB não seja enviado, pois o `.gitignore` já o protege).*

### Passo 2: Publicar no Streamlit Community Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io/) e faça login com sua conta do GitHub.
2. Clique no botão **"New app"** (ou "Create app").
3. Selecione o repositório que você acabou de criar:
   - **Repository:** `SEU_USUARIO/app-predicao-empregabilidade`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Clique em **"Deploy!"**.
5. O Streamlit Cloud instalará as dependências do `requirements.txt` e sua aplicação ficará disponível mundialmente com uma URL pública (ex.: `https://predicao-empregabilidade.streamlit.app`).

---

## 📊 Onde Colocar o Dataset? (Dúvida Frequente)

* **No GitHub e no Streamlit Cloud:**  
  **Você NÃO precisa subir o dataset CSV.** O Streamlit Cloud executa apenas o arquivo compilado `modelo/modelo_emprego.pkl`, que pesa menos de 100 KB. O GitHub rejeita arquivos maiores que 100 MB, e o CSV do Stack Overflow possui mais de 120 MB.
* **No seu computador local (para re-treinamento):**  
  Se você quiser rodar `python treinamento.py`, basta salvar o arquivo `survey_results_public.csv` dentro da pasta `dataset/`. O script identificará automaticamente e gerará um novo arquivo `.pkl` na pasta `modelo/`.
