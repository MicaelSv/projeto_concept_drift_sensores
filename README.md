# Concept drift na classificação de gases

Código e protocolo experimental do trabalho **“Impacto do Concept Drift na Classificação de Gases: Uma Avaliação Temporal de Estratégias de Atualização de Modelos”**.

O projeto avalia como a deriva temporal dos sensores afeta a identificação de seis gases e compara cinco políticas de treinamento: modelo estático no primeiro lote, último lote, janela móvel de três lotes, treinamento cumulativo e ensemble ponderado por recência. A análise é repetida com SVM de kernel RBF e Random Forest.

## Dados

Foi utilizado o [Gas Sensor Array Drift Dataset at Different Concentrations, da UCI](https://archive.ics.uci.edu/dataset/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset). A base contém 13.910 medições, 128 atributos extraídos de 16 sensores químicos, seis classes e dez lotes em ordem temporal.

| Classe | Gás |
|---:|---|
| 1 | Etanol |
| 2 | Etileno |
| 3 | Amônia |
| 4 | Acetaldeído |
| 5 | Acetona |
| 6 | Tolueno |

## Como executar localmente

Recomenda-se Python 3.11 ou mais recente.

1. Clone o repositório e entre na pasta:

   ```bash
   git clone URL_DO_REPOSITORIO
   cd projeto_concept_drift_sensores
   ```

2. Crie e ative um ambiente virtual no Windows PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Baixe os dez arquivos `batch1.dat` a `batch10.dat` na página da UCI indicada acima e coloque-os em `data/raw/`.

4. Confira a integridade e consolide os lotes:

   ```powershell
   python scripts/preparar_dados.py
   ```

5. Execute o experimento final:

   ```powershell
   python scripts/validacao_classificadores_artigo.py
   ```

O experimento respeita a ordem B1–B10: para avaliar o lote `Bt`, somente lotes anteriores são usados no treinamento. A métrica principal é o Macro-F1 calculado com as seis classes fixas. A Random Forest é executada com cinco sementes; a SVM é determinística no protocolo adotado.

## Saídas geradas

Ao final, os seguintes artefatos são criados localmente:

- `reports/figures/validacao_classificadores_macro_f1.png`: figura principal do artigo;
- `reports/tables/resumo_validacao_classificadores.csv`: resumo por classificador e estratégia;
- `reports/tables/comparacao_classificadores_por_lote.csv`: métricas de cada lote;
- `reports/tables/f1_por_classe_validacao_classificadores.csv`: F1 por gás;
- `reports/tables/testes_pareados_validacao_classificadores.csv`: testes de Wilcoxon com correção de Holm;
- `results/resultados_validacao_classificadores.csv`: resultados completos por semente e lote.

Esses diretórios são ignorados pelo Git para evitar o versionamento de artefatos gerados. Os arquivos podem ser reproduzidos com os comandos acima.

## Principais resultados

O treinamento cumulativo obteve o melhor Macro-F1 médio nos dois classificadores:

| Classificador | Cumulativo | Janela de 3 lotes | Ensemble | Último lote | Estático B1 |
|---|---:|---:|---:|---:|---:|
| SVM RBF | **0,724** | 0,705 | 0,616 | 0,612 | 0,353 |
| Random Forest | **0,693** | 0,685 | 0,636 | 0,621 | 0,370 |

Os resultados indicam degradação temporal do modelo estático e maior estabilidade quando o treinamento reutiliza o histórico. O estudo não implementa um detector formal de drift; ele avalia políticas de atualização em uma sequência temporal já definida.

## Estrutura principal

```text
config/       protocolos e hiperparâmetros
checksums/    manifestos SHA-256 para rastreabilidade
data/         dados locais ignorados pelo Git
scripts/      preparação dos dados e experimento final
reports/      figuras e tabelas geradas localmente
results/      resultados detalhados gerados localmente
```

## Reprodutibilidade

- Os arquivos originais são conferidos pelo manifesto `checksums/originais.sha256`.
- A consolidação cria um manifesto local para validar as entradas usadas no experimento.
- Os hiperparâmetros estão em `config/protocolo_validacao_classificadores.json`.
- Não há ajuste de hiperparâmetros usando os lotes de teste.

Notebooks (`*.ipynb`), arquivos compactados (`*.zip`), dados brutos e artefatos gerados não fazem parte do repositório.
