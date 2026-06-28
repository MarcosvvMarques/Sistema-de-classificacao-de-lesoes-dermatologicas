# Classificação de Lesões Dermatológicas com Deep Learning

> **Entrega 1 — Proposta Formal** · 30/jun/2026 · Módulo 7 — Projetos de IA (PPI/SiDi)

---

## 1. Introdução [E1]

### 1.1 Título

Classificação de Lesões Dermatológicas: EfficientNet-B3 vs. ResNet18 com Explicabilidade via Grad-CAM

### 1.2 Equipe

| Nome | E-mail SiDi | Papel principal | Contribuição principal (E2) |
|------|-------------|-----------------|------------------------------|
| [TODO] | [TODO] | Líder de equipe | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de dados | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de modelagem | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de engenharia | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de avaliação | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de documentação | [TODO Entrega 2] |
| [TODO] | [TODO] | Líder de apresentação | [TODO Entrega 2] |

### 1.3 Contexto e motivação

O câncer de pele é o tipo mais comum de câncer no Brasil e no mundo. A detecção precoce é o principal fator prognóstico: melanomas diagnosticados em estágios iniciais têm taxa de sobrevida em 5 anos superior a 98%, enquanto casos avançados chegam a menos de 30%. O diagnóstico dermatoscópico, porém, é altamente especializado e os dermatologistas ainda são distribuídos de forma desigual no território brasileiro — regiões do interior e do Norte/Nordeste possuem cobertura muito inferior à necessária.

Modelos de classificação de imagens baseados em deep learning demonstraram desempenho comparável a dermatologistas especializados em estudos como Esteva et al. (2017, *Nature*) e Haenssle et al. (2018, *Annals of Oncology*). Este projeto aplica transfer learning sobre o dataset público HAM10000 para construir um sistema de apoio ao diagnóstico que classifica 7 categorias de lesões dermatológicas, acompanhado de mapas de ativação Grad-CAM que explicam quais regiões anatômicas foram decisivas para cada predição — tornando o sistema auditável por clínicos.

### 1.4 Problema / pergunta de pesquisa

> É possível, com transfer learning sobre EfficientNet-B3 pré-treinado em ImageNet, atingir AUC-ROC ≥ 0,85 para a classe melanoma no conjunto de teste do HAM10000, superando o baseline ResNet18?

### 1.5 Hipótese

O EfficientNet-B3 fine-tuned com balanceamento de classes por pesos e data augmentation agressivo atingirá AUC-ROC ≥ 0,85 para melanoma e F1-macro ≥ 0,60 sobre todas as 7 classes, superando o ResNet18 baseline com margem estatisticamente significativa.

### 1.6 Objetivos

1. Preparar um pipeline reprodutível de ingestão, organização e pré-processamento do HAM10000.
2. Conduzir EDA rigorosa identificando desbalanceamento de classes, distribuições demográficas e possíveis vieses.
3. Treinar ResNet18 como baseline e EfficientNet-B3 como modelo principal, comparando-os com métricas padronizadas.
4. Implementar GradCAM e GradCAM++ para gerar mapas de ativação sobrepostos sobre as predições.
5. Documentar o projeto com README executável e código reprodutível end-to-end.

---

## 2. Dados

### 2.1 Fonte e licença [E1]

| Campo | Valor |
|-------|-------|
| Dataset | HAM10000 (Human Against Machine with 10000 training images) |
| Fonte | Kaggle — `kmader/skin-cancer-mnist-ham10000` |
| URL | https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection |
| Licença | **CC BY-NC 4.0** — permite uso acadêmico, proíbe uso comercial |
| Data de acesso | Maio/2026 |
| Autenticação | Requer conta Kaggle + `kaggle.json` ou variáveis de ambiente |

### 2.2 Volume e formato [E1]

| Campo | Valor |
|-------|-------|
| Total de imagens | 10.015 |
| Lesões únicas | ~7.470 (algumas lesões têm múltiplas imagens) |
| Formato | JPEG, RGB |
| Resolução original | 600 × 450 px (redimensionadas para 224 × 224 no pipeline) |
| Tamanho em disco | ~3 GB |
| Metadados | CSV com `image_id`, `lesion_id`, `dx`, `dx_type`, `age`, `sex`, `localization` |

### 2.3 Variáveis principais [E1]

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `image_id` | string | ID único da imagem (ex.: ISIC_0024306) |
| `lesion_id` | string | ID da lesão (várias imagens podem compartilhar a mesma lesão) |
| `dx` | categórica (7 classes) | Diagnóstico: mel, nv, bcc, akiec, bkl, df, vasc |
| `dx_type` | categórica | Método de diagnóstico: histo, follow_up, consensus, confocal |
| `age` | numérica | Idade do paciente em anos |
| `sex` | categórica | Sexo biológico: male, female |
| `localization` | categórica | Localização da lesão no corpo (18 valores) |

### 2.4 Riscos de dados [E1]

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| **Desbalanceamento severo** | Nevo (nv) representa ~67% dos dados; vasc e df < 1,5% cada | WeightedRandomSampler + pesos de classe na loss |
| **Data leakage** | Múltiplas imagens da mesma lesão podem cair em splits diferentes | Split feito ao nível de `lesion_id` (paciente), não de imagem |
| **Viés demográfico** | Dataset predominantemente de pacientes europeus com pele clara | Limitação declarada; modelo não deve ser usado em populações muito distintas sem re-calibração |
| **Ruído de anotação** | Diagnósticos por consenso clínico podem ter ambiguidade | Usamos rótulos como fornecidos; dx_type registrado |
| **PII** | Apenas idade e sexo — sem nome, CPF ou identificação direta | Dados já anonimizados pelo provedor ISIC |

### 2.5 Pré-processamento aplicado [E2]

[TODO Entrega 2] Descrever etapas efetivamente aplicadas após experimentos.

### 2.6 Ética e privacidade [E1]

- Dados anonimizados pelo ISIC Archive; nenhuma identificação pessoal além de idade e sexo.
- Licença CC BY-NC 4.0 permite uso acadêmico irrestrito.
- Modelo é concebido como **ferramenta de apoio** ao diagnóstico — não substitui avaliação médica.
- Limitação de representatividade (dataset majoritariamente de pele clara) será declarada explicitamente nos resultados.

---

## 3. Metodologia

### 3.1 Abordagem [E1]

Classificação supervisionada multiclasse de imagens dermatoscópicas com deep learning. Usaremos transfer learning — pesos pré-treinados no ImageNet são adaptados ao domínio dermatológico via fine-tuning completo com taxa de aprendizado diferenciada (menor nas camadas iniciais).

### 3.2 Stack técnica [E1]

| Componente | Ferramenta |
|------------|-----------|
| Linguagem | Python 3.10+ |
| Deep learning | PyTorch ≥ 2.1, timm ≥ 0.9 |
| Augmentation | Albumentations ≥ 1.3 |
| Métricas / EDA | scikit-learn, pandas, matplotlib, seaborn |
| Visualização | OpenCV (Grad-CAM overlay) |
| Reprodutibilidade | seeds fixas (42), `requirements.txt` |
| Gestão de dados | kagglehub + python-dotenv |

### 3.3 Baselines [E1]

| Baseline | Descrição |
|----------|-----------|
| **Majoritário** | Sempre prediz "nv" (classe mais frequente); acurácia teórica ~67% mas AUC = 0,50 |
| **ResNet18** | ResNet-18 pré-treinado em ImageNet, fine-tuned com mesmo pipeline de augmentation |

O ResNet18 é o baseline de referência do projeto. A comparação com o EfficientNet-B3 demonstrará o ganho obtido por uma arquitetura mais eficiente.

### 3.4 Pipeline [E2]

[TODO Entrega 2]

```
Dados brutos → Limpeza/Split → Augmentation → Modelo → Avaliação → Grad-CAM
```

### 3.5 Modelos comparados [E2]

[TODO Entrega 2]

### 3.6 Protocolo de validação [E1]

| Parâmetro | Valor |
|-----------|-------|
| Split strategy | Estratificado por classe, ao nível de `lesion_id` |
| Proporção | 70% treino / 15% validação / 15% teste |
| Semente global | `SEED = 42` (fixada em random, numpy, torch) |
| Vazamento | Impossível — uma lesão nunca aparece em dois splits |
| Métricas de parada | Melhor `val_acc` com early stopping (patience = 10 épocas) |

### 3.7 Métricas e critérios de sucesso [E1]

| Métrica | Justificativa | Valor mínimo aceitável |
|---------|---------------|------------------------|
| **AUC-ROC melanoma** | Meta primária do projeto; robusto a desbalanceamento | **≥ 0,85** |
| **AUC-ROC macro** | Desempenho médio sobre todas as 7 classes | ≥ 0,80 |
| **F1-macro** | Penaliza classes com baixo recall (classes raras) | ≥ 0,55 |
| **Acurácia balanceada** | Não enviesada pela classe majoritária | ≥ 0,65 |

---

## 4. Cronograma [E1 planejado; E2 status]

| Semana | Período | Marco / atividade | Status (E2) |
|--------|---------|-------------------|-------------|
| 1 | 29/mai–05/jun | Aquisição de dados + EDA inicial | [TODO E2] |
| 2 | 06–12/jun | EDA aprofundada + definição de features de augmentation | [TODO E2] |
| 3 | 13–19/jun | Baseline ResNet18 + linha de base de métricas | [TODO E2] |
| 4 | 20–26/jun | EfficientNet-B3 v1 + experimentos iniciais | [TODO E2] |
| 5 | 27/jun–03/jul | **Entrega 1 (30/jun)** + refinamento pós-feedback | [TODO E2] |
| 6 | 04–10/jul | Tuning de hiperparâmetros + ablation studies | [TODO E2] |
| 7 | 11–17/jul | Grad-CAM + análise de erros + fairness | [TODO E2] |
| 8 | 18–24/jul | Documentação final + ensaio de demo | [TODO E2] |
| 9 | 25–31/jul | **Entrega 2 (31/jul)** — Seminário + slides + defesa | [TODO E2] |

**Aprendizado sobre desvios (E2):** [TODO Entrega 2]

---

## 5. Resultados [E2]

### 5.1 Métricas obtidas

[TODO Entrega 2]

| Modelo | AUC-ROC mel | AUC-ROC macro | F1-macro | Acurácia |
|--------|-------------|---------------|----------|----------|
| Majoritário | 0,500 | — | — | ~67% |
| ResNet18 (baseline) | [TODO] | [TODO] | [TODO] | [TODO] |
| EfficientNet-B3 | [TODO] | [TODO] | [TODO] | [TODO] |

### 5.2 Gráficos relevantes

[TODO Entrega 2] — ver `docs/figs/`

### 5.3 Análise de erros

[TODO Entrega 2]

### 5.4 Comparação com a hipótese

[TODO Entrega 2]

---

## 6. Conclusão [E2]

### 6.1 Principais achados

[TODO Entrega 2]

### 6.2 Limitações

[TODO Entrega 2]

### 6.3 Trabalhos futuros

[TODO Entrega 2]

### 6.4 Aprendizados da equipe

[TODO Entrega 2]

---

## 7. Reprodutibilidade [E2]

### 7.1 Requisitos

- Python 3.10+
- CUDA 11.8+ (opcional; CPU funciona com velocidade reduzida)
- Conta Kaggle com credenciais configuradas
- Sistema testado: Ubuntu 22.04 / Linux

Semente global: `42`

### 7.2 Instalação

```bash
git clone <URL-do-repositório>
cd Sistema-de-classificacao-de-lesoes-dermatologicas

# Recomendado: ambiente isolado
python -m venv venv && source venv/bin/activate

pip install -r requirements.txt
```

### 7.3 Obter os dados

```bash
# Configure credenciais Kaggle
cp .env.example .env   # edite com KAGGLE_USERNAME e KAGGLE_KEY

make download          # baixa o dataset (~3 GB)
make prepare           # organiza em train/val/test por classe
make eda               # gera figuras EDA em docs/figs/
```

### 7.4 Executar o pipeline

```bash
make train-baseline    # ResNet18
make train-main        # EfficientNet-B3
make compare           # tabela comparativa de métricas
make gradcam           # mapas Grad-CAM++ sobre o conjunto de teste
```

Ou pipeline completo:

```bash
make all
```

### 7.5 Artefatos gerados

| Diretório | Conteúdo |
|-----------|----------|
| `checkpoints/` | Pesos `.pth` dos melhores modelos por epoch de validação |
| `outputs/evaluation/` | Matriz de confusão e curvas ROC por modelo |
| `outputs/gradcam/` | Imagens com overlay Grad-CAM++ |
| `outputs/comparison/` | Gráfico comparativo entre modelos |
| `docs/figs/` | Figuras da EDA |
