# Pasta do Modelo

Esta pasta armazena o artefato binário do modelo de Machine Learning (`pickle`).

## Arquivo esperado:
- `modelo_emprego.pkl` (ou `modelo_emprego.sav`)

### Como obter o arquivo:
1. **Pelo Google Colab:**
   - No seu notebook executado no Google Colab, o modelo foi salvo no seu Google Drive em `/content/drive/MyDrive/Lab_IA/Dataset/modelo_emprego.sav`.
   - Baixe esse arquivo do Google Drive para o seu computador.
   - Renomeie-o ou salve-o aqui como `modelo_emprego.pkl` (ou mantenha `modelo_emprego.sav`).
2. **Executando o script local de treino:**
   - Com o dataset presente em `dataset/survey_results_public.csv`, execute:
     ```bash
     python treinamento.py
     ```
   - O arquivo `modelo_emprego.pkl` será automaticamente gerado nesta pasta.
