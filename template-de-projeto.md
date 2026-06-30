# [Titulo do Projeto]

> **Template único** dos projetos de IA do Módulo 7 (PPI/SiDi). Este mesmo arquivo é usado em duas fases:
>
> - **Entrega 1 — Proposta (30/jun/2026):** preencha todas as seções marcadas com `[E1]`. As seções `[E2]` ficam intocadas (mantenha os `[TODO]`).
> - **Entrega 2 — Versão final / README do repositório (31/jul/2026):** complete as seções `[E2]` e revise as `[E1]` para refletir o que efetivamente foi feito (ajuste objetivos, dados ou metodologia se mudou de rumo, e justifique).
>
> Substitua todos os `[TODO]`. Os blocos `<!-- exemplo: ... -->` são apenas referência e devem ser apagados após preencher.
>
> Na Entrega 2, este arquivo passa a ser o `README.md` do repositório do projeto: um avaliador deve conseguir entender, instalar e executar o projeto lendo apenas este documento.

---

## 1. Introdução `[E1]`

### 1.1 Título

[TODO] Título curto, informativo e memorável (até 80 caracteres). Evite jargão.

### 1.2 Equipe

| Nome   | E-mail SiDi | Papel principal | Contribuição principal (preencher na E2) |
|--------|-------------|-----------------|------------------------------------------|
| [TODO] | [TODO]      | [TODO]          | [TODO Entrega 2]                         |

- **Tamanho da equipe:** 6 a 7 alunos (a turma de 62 alunos será dividida em 10 grupos).
- **Papéis sugeridos** (cada integrante assume um papel principal; espera-se colaboração entre todos):
  - **Líder de equipe** — coordena planejamento, comunicação com o instrutor e divisão de tarefas.
  - **Líder de dados** — coleta, limpeza e EDA (área: Ciência de Dados).
  - **Líder de modelagem** — treino, validação e comparação de modelos (área: Machine Learning).
  - **Líder de engenharia** — pipelines reprodutíveis e infraestrutura (área: Engenharia de Dados).
  - **Líder de avaliação** — métricas, análise de erros e fairness.
  - **Líder de documentação** — README, relatórios e versionamento da entrega.
  - **Líder de apresentação** — slides, narrativa e defesa no seminário.

> *Para grupos de 6 alunos, um dos papéis pode ser acumulado (ex.: Líder de equipe + apresentação).*

### 1.3 Contexto e motivação

[TODO] 1-2 parágrafos: domínio, por que o problema importa, quem se beneficia da solução.

### 1.4 Problema / pergunta de pesquisa

[TODO] Uma única pergunta ou objetivo operacional claro. Evite objetivos múltiplos.

### 1.5 Hipótese

[TODO] Hipótese testável. Relacione a variável preditora com a resposta esperada.

### 1.6 Objetivos

[TODO] Lista concisa (3-5 itens). Na Entrega 2, revise para refletir os objetivos efetivamente perseguidos; se mudaram em relação à proposta, explique por quê.

---

## 2. Dados

### 2.1 Fonte e licença `[E1]`

[TODO] URL do dataset, licença, data de acesso, versão/snapshot. Inclua se requer autenticação.

### 2.2 Volume e formato `[E1]`

[TODO] Número de amostras, colunas, tamanho em disco, formato (CSV, Parquet, JSON, imagens), periodicidade.

### 2.3 Variáveis principais `[E1]`

[TODO] Tabela com as 5-10 variáveis mais importantes.

| Variável | Tipo   | Descrição |
|----------|--------|-----------|
| [TODO]   | [TODO] | [TODO]    |

### 2.4 Riscos de dados `[E1]`

[TODO] PII? Dados sensíveis? Desbalanceamento? Ruído? Drift? Como serão tratados?

### 2.5 Pré-processamento aplicado `[E2]`

[TODO Entrega 2] Etapas de limpeza, feature engineering, normalização, split train/test, tratamento de desbalanceamento. Compare com o que foi previsto em 2.4.

### 2.6 Ética e privacidade `[E1, revisar na E2]`

[TODO] Dados contêm PII? Foi anonimizado? Licença permite uso acadêmico?

---

## 3. Metodologia

### 3.1 Abordagem `[E1]`

[TODO] Supervisionado / não-supervisionado / híbrido? Classificação / regressão / clustering / deep learning? Justifique a escolha em função do problema e dos dados.

### 3.2 Stack técnica `[E1]`

[TODO] Linguagem, libs, ambientes (ex.: Python 3.11, scikit-learn, PyTorch, MLflow). Alinhe com a stack das aulas.

### 3.3 Baselines `[E1]`

[TODO] Qual baseline trivial será comparado? (ex.: sempre prever classe majoritária, regressão logística simples).

### 3.4 Pipeline `[E2]`

[TODO Entrega 2] Diagrama ou descrição passo a passo: raw data → limpeza → features → modelo → avaliação.

```mermaid
flowchart LR
    A[Dados brutos] --> B[Limpeza]
    B --> C[Feature engineering]
    C --> D[Modelo]
    D --> E[Avaliação]
```

### 3.5 Modelos comparados `[E2]`

[TODO Entrega 2] Liste os modelos efetivamente avaliados (baseline + candidatos). Referencie hiperparâmetros principais.

### 3.6 Protocolo de validação `[E1]`

[TODO] Train/test split? K-fold? Time-based? Semente fixa? Como garantirá (ou garantiu) que não haverá (houve) vazamento?

### 3.7 Métricas e critérios de sucesso `[E1: planejados; E2: aplicados]`

[TODO] Cite 2-4 métricas com justificativa. Para cada uma, defina o **valor mínimo aceitável** já na Entrega 1. Na Entrega 2, mantenha esses valores e use-os como referência para a Seção 5 (Resultados).

---

## 4. Cronograma `[E1: planejado; E2: status]`

A janela real de implementação é de ~9 semanas (29/mai → 31/jul/2026), com 2 marcos formais. Ajuste as atividades semanais ao escopo do seu projeto. Na Entrega 2, atualize a coluna **Status** marcando cada semana como `[OK]`, `[Atrasado]` ou `[Não feito]`, e adicione 1 linha com aprendizado sobre desvios.

| Semana | Período        | Marco / atividade prevista                                          | Status (E2) |
|--------|----------------|---------------------------------------------------------------------|-------------|
| 1      | 29/mai–05/jun  | [TODO] Pré-especificação + aquisição de dados + EDA inicial         | [TODO E2]   |
| 2      | 06–12/jun      | [TODO] EDA aprofundada + definição de features                      | [TODO E2]   |
| 3      | 13–19/jun      | [TODO] Baseline trivial e linha de base de métricas                 | [TODO E2]   |
| 4      | 20–26/jun      | [TODO] Modelo principal v1 + experimentos iniciais                  | [TODO E2]   |
| 5      | 27/jun–03/jul  | **Entrega 1 (30/jun)** + refinamento pós-feedback                  | [TODO E2]   |
| 6      | 04–10/jul      | [TODO] Tuning + ablation studies                                    | [TODO E2]   |
| 7      | 11–17/jul      | [TODO] Análise de erros + fairness (se aplicável)                   | [TODO E2]   |
| 8      | 18–24/jul      | [TODO] Documentação final (preencher seções E2) + ensaio de demo   | [TODO E2]   |
| 9      | 25–31/jul      | **Entrega 2 (31/jul)** — Seminário + slides + defesa               | [TODO E2]   |

> *Os marcos das semanas 5 e 9 são fixos; as demais atividades podem ser realocadas conforme a complexidade do tema escolhido.*

**Aprendizado sobre desvios (E2):** [TODO Entrega 2] 2-3 linhas sobre o que mudou em relação ao plano e por quê.

---

## 5. Resultados `[E2]`

### 5.1 Métricas obtidas

[TODO Entrega 2] Tabela com baseline e modelo principal. Compare com os valores mínimos definidos em 3.7.

| Modelo           | Métrica 1 | Métrica 2 | Métrica 3 |
|------------------|-----------|-----------|-----------|
| [TODO baseline]  | [TODO]    | [TODO]    | [TODO]    |
| [TODO principal] | [TODO]    | [TODO]    | [TODO]    |

### 5.2 Gráficos relevantes

[TODO Entrega 2] Matriz de confusão, curva PR, curva ROC, SHAP etc. Inclua ou referencie imagens em `docs/figs/`.

### 5.3 Análise de erros

[TODO Entrega 2] Onde o modelo falha? Casos sistemáticos de falso positivo/negativo?

### 5.4 Comparação com a hipótese

[TODO Entrega 2] A hipótese declarada em 1.5 foi confirmada? Parcialmente? Explique com evidências.

---

## 6. Conclusão `[E2]`

### 6.1 Principais achados

[TODO Entrega 2] 3-5 pontos concretos que o projeto demonstrou.

### 6.2 Limitações

[TODO Entrega 2] O que não foi feito, o que não generaliza, onde o projeto não deveria ser aplicado.

### 6.3 Trabalhos futuros

[TODO Entrega 2] O que você faria com mais 6 semanas?

### 6.4 Aprendizados da equipe

[TODO Entrega 2] O que a equipe aprendeu (técnico e de processo)?

---

## 7. Reproducibilidade `[E2]`

### 7.1 Requisitos

- Python [TODO Entrega 2 versão, ex.: 3.11+]
- [TODO Entrega 2 lista de libs principais ou referência ao `pyproject.toml`]
- Sistema operacional testado: [TODO Entrega 2]
- Semente global: [TODO Entrega 2, ex.: 42]

### 7.2 Instalação

```bash
# Clonar o repositório
git clone [TODO URL]
cd [TODO nome-do-projeto]

# Instalar dependências com uv (recomendado)
uv sync

# Alternativa com pip
# pip install -r requirements.txt
```

### 7.3 Obter os dados

```bash
# [TODO Entrega 2: comando de download ou instrução para baixar manualmente]
# Ex.: make download-data
```

### 7.4 Executar o pipeline

```bash
# [TODO Entrega 2: comando único ou sequência documentada]
# Ex.:
# make train
# make evaluate
# make report
```

### 7.5 Rodar os testes (se aplicável)

```bash
# [TODO Entrega 2]
# Ex.: pytest -v
```

### 7.6 Artefatos gerados

[TODO Entrega 2] Liste o que será gerado em `artifacts/`, `models/`, `reports/`. Tamanho esperado.
