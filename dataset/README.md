# Pasta do Dataset

## ⚠️ Por que o dataset completo (`survey_results_public.csv`) não sobe para o GitHub?

1. **Limite do GitHub:** O GitHub possui um limite estrito de **100 MB** por arquivo. O arquivo do Stack Overflow possui mais de **120 MB**, o que causa erro de recusa de upload no Git (`remote: error: GH001: Large files detected`).
2. **O Streamlit Cloud NÃO precisa do arquivo CSV para rodar:**
   - A aplicação web (`app.py`) apenas faz **inferências** (predições em tempo real).
   - Ela utiliza apenas o arquivo compilado do modelo treinado localizado em `modelo/modelo_emprego.pkl` (que tem apenas alguns kilobytes e sobe perfeitamente para o GitHub).

---

## Onde colocar o dataset localmente se você quiser re-treinar o modelo?

Se você quiser rodar o script `treinamento.py` no seu computador:
1. Baixe o arquivo `survey_results_public.csv` do Stack Overflow Developer Survey (ou utilize o que você já baixou).
2. Coloque o arquivo `survey_results_public.csv` diretamente dentro desta pasta `dataset/`:
   ```
   app_Predicao_Empregabilidade/
   └── dataset/
       └── survey_results_public.csv   <-- Coloque aqui
   ```
3. O arquivo `.gitignore` já está configurado para **ignorar** qualquer `.csv` dentro desta pasta, garantindo que você nunca envie acidentalmente mais de 100 MB para o GitHub.
