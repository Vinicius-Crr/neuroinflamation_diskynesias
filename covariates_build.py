# =============================================================================
# Construção do arquivo de covariáveis para análise de associação genética
# =============================================================================
# Projeto  : Análise genética de discinesia na Doença de Parkinson
# Dataset  : AMP-PD 2023_v4release_1027
# Ambiente : Terra/Google Cloud (Jupyter Notebook)
# Autor    : [seu nome]
# Data     : [data]
#
# Descrição:
#   Este script constrói o arquivo de covariáveis (covariables_restrict_final.txt)
#   necessário para a análise de associação genética (GWAS).
#
#   O pipeline segue as seguintes etapas:
#     1. Configuração do ambiente e instalação de ferramentas
#     2. Download dos arquivos genômicos e clínicos do GCS
#     3. Construção do arquivo de fenótipos (casos/controles)
#     4. Integração com dados clínicos (demographics, UPDRS, histórico médico)
#     5. Filtragem de amostras por critérios clínicos e de qualidade
#     6. Cálculo de componentes principais (PCA) para controle de estratificação
#     7. Remoção de outliers de PCA
#     8. Geração do arquivo final de covariáveis com PCs
#
# Saída:
#   covariables_restrict_final.txt — arquivo TSV com FID, IID, PHENO e covariáveis
#
# Nota sobre nomenclatura ("restrict"):
#   O sufixo "restrict" refere-se ao critério restrito de inclusão:
#   somente participantes em uso de levodopa (on_levodopa == 'Yes'),
#   raça branca, etnia não-hispânica, e com idades biologicamente plausíveis.
# =============================================================================


# =============================================================================
# SEÇÃO 1 — IMPORTAÇÕES E FUNÇÕES UTILITÁRIAS
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from functools import reduce

from google.cloud import bigquery
from firecloud import api as fapi
from IPython.display import display, HTML
import urllib.parse

%matplotlib inline


# ── Funções auxiliares do ambiente Terra ─────────────────────────────────────

def shell_do(command):
    """Imprime e executa um comando de shell."""
    print(f'Executando: {command}', file=sys.stderr)
    !$command

def shell_return(command):
    """Executa um comando de shell e retorna o output como string."""
    print(f'Executando: {command}', file=sys.stderr)
    output = !$command
    return '\n'.join(output)

def bq_query(query):
    """Executa uma query no BigQuery e retorna um DataFrame."""
    print(f'Executando: {query}', file=sys.stderr)
    return pd.read_gbq(query, project_id=BILLING_PROJECT_ID, dialect='standard')

def gcs_read_file(path):
    """Lê o conteúdo de um arquivo no Google Cloud Storage."""
    contents = !gsutil -u {BILLING_PROJECT_ID} cat {path}
    return '\n'.join(contents)

def gcs_read_csv(path, sep=None):
    """Lê um arquivo delimitado do GCS e retorna um DataFrame."""
    return pd.read_csv(StringIO(gcs_read_file(path)), sep=sep, engine='python')


# =============================================================================
# SEÇÃO 2 — VARIÁVEIS DO AMBIENTE TERRA
# =============================================================================

# Variáveis injetadas automaticamente pelo Terra
BILLING_PROJECT_ID = os.environ['GOOGLE_PROJECT']
WORKSPACE_NAMESPACE = os.environ['WORKSPACE_NAMESPACE']
WORKSPACE_NAME      = os.environ['WORKSPACE_NAME']
WORKSPACE_BUCKET    = os.environ['WORKSPACE_BUCKET']

WORKSPACE_ATTRIBUTES = (
    fapi.get_workspace(WORKSPACE_NAMESPACE, WORKSPACE_NAME)
    .json()
    .get('workspace', {})
    .get('attributes', {})
)

print('── Workspace ──────────────────────────────')
print(f'Nome     : {WORKSPACE_NAME}')
print(f'Projeto  : {BILLING_PROJECT_ID}')
print(f'Bucket   : {WORKSPACE_BUCKET}')

# Caminhos do AMP-PD v4 no GCS
AMP_RELEASE_PATH          = 'gs://amp-pd-data/releases/2023_v4release_1027'
AMP_CLINICAL_RELEASE_PATH = f'{AMP_RELEASE_PATH}/clinical'
AMP_WGS_RELEASE_PATH      = 'gs://amp-pd-genomics/releases/2023_v4release_1027'

print('\n── AMP-PD v4 ──────────────────────────────')
print(f'Clínico  : {AMP_CLINICAL_RELEASE_PATH}')
print(f'WGS      : {AMP_WGS_RELEASE_PATH}')

# Bucket seguro do workspace (onde ficam os arquivos do projeto)
PROJECT_BUCKET = 'gs://fc-secure-5dd0ff90-839e-485e-9021-a5a7ad3060ec'

# Diretório local de trabalho
WORK_DIR = 'data'
shell_do(f'mkdir -p {WORK_DIR}')


# =============================================================================
# SEÇÃO 3 — INSTALAÇÃO DO PLINK 1.9 E PLINK 2.0
# =============================================================================

# PLINK 1.9 — usado para operações legadas se necessário
!wget -q -O plink1.9.zip https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20201019.zip
!unzip -q plink1.9.zip -d plink1.9_folder

# PLINK 2.0 — usado para QC, LD pruning e PCA
!wget -q -O plink2.zip https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20241124.zip
!unzip -q plink2.zip -d plink2_folder

# Adiciona os executáveis ao PATH da sessão
os.environ["PATH"] += os.pathsep + os.path.abspath("./plink1.9_folder")
os.environ["PATH"] += os.pathsep + os.path.abspath("./plink2_folder")


# =============================================================================
# SEÇÃO 4 — DOWNLOAD DOS ARQUIVOS DO GCS
# =============================================================================

# ── Dados clínicos do AMP-PD ─────────────────────────────────────────────────
arquivos_clinicos = [
    'clinical/Demographics.csv',
    'clinical/Demographics_dictionary.csv',
    'clinical/Enrollment.csv',
    'clinical/Enrollment_dictionary.csv',
    'clinical/PD_Medical_History.csv',
    'clinical/PD_Medical_History_dictionary.csv',
    'clinical/MDS_UPDRS_Part_III.csv',
    'clinical/MDS_UPDRS_Part_IV.csv',
]
for arq in arquivos_clinicos:
    shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {AMP_RELEASE_PATH}/{arq} {WORK_DIR}')

# Inventário de amostras WGS
shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {AMP_RELEASE_PATH}/wgs_WB-DWGS_sample_inventory.csv {WORK_DIR}')

# ── Amostras selecionadas via SQL (BigQuery) ──────────────────────────────────
# Gerados em 01_sample_selection/
shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {PROJECT_BUCKET}/CASOS_06_08.csv {WORK_DIR}')
shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {PROJECT_BUCKET}/CONTROLES_06_08.csv {WORK_DIR}')

# ── Arquivo genômico mesclado (todos os genes, pré-filtrado) ──────────────────
for ext in ['pgen', 'psam', 'pvar']:
    shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {PROJECT_BUCKET}/todos_genes_filtrado.{ext} {WORK_DIR}')


# =============================================================================
# SEÇÃO 5 — LEITURA DOS ARQUIVOS CLÍNICOS
# =============================================================================

# Dados demográficos
demo     = pd.read_csv(f'{WORK_DIR}/Demographics.csv')

# Dados de matrícula no estudo
enrol    = pd.read_csv(f'{WORK_DIR}/Enrollment.csv')

# Histórico médico — uma linha por participante (remove duplicatas)
med      = pd.read_csv(f'{WORK_DIR}/PD_Medical_History.csv')
med      = med.drop_duplicates(subset='participant_id')

# MDS-UPDRS Parte III (motor) e Parte IV (complicações motoras)
UPDRSIII = pd.read_csv(f'{WORK_DIR}/MDS_UPDRS_Part_III.csv')
UPDRSIV  = pd.read_csv(f'{WORK_DIR}/MDS_UPDRS_Part_IV.csv')

# Inventário de amostras WGS — contém o sample_id usado pelo PLINK
sample_inventory = pd.read_csv(f'{WORK_DIR}/wgs_WB-DWGS_sample_inventory.csv')

# Amostras selecionadas no BigQuery
casos      = pd.read_csv(f'{WORK_DIR}/CASOS_06_08.csv')
controles  = pd.read_csv(f'{WORK_DIR}/CONTROLES_06_08.csv')


# =============================================================================
# SEÇÃO 6 — SELEÇÃO DA VISITA MAIS RECENTE (UPDRS III e IV)
# =============================================================================
# Os participantes têm múltiplas visitas. Para covariáveis, usamos
# a visita mais recente disponível de cada escala.

III_recente = (
    UPDRSIII
    .sort_values(['participant_id', 'visit_month'], ascending=[True, False])
    .groupby('participant_id')
    .last()
    .reset_index()
)

IV_recente = (
    UPDRSIV
    .sort_values(['participant_id', 'visit_month'], ascending=[True, False])
    .groupby('participant_id')
    .last()
    .reset_index()
)


# =============================================================================
# SEÇÃO 7 — CONSTRUÇÃO DO ARQUIVO DE FENÓTIPOS (PLINK FORMAT)
# =============================================================================
# O PLINK espera um arquivo com colunas FID, IID, PHENO
# Convenção: 1 = controle, 2 = caso

# Vincula participant_id → sample_id (identificador genômico)
casos_merged     = pd.merge(casos,     sample_inventory, on='participant_id')
controles_merged = pd.merge(controles, sample_inventory, on='participant_id')

# Atribui código de fenótipo
casos_merged['PHENO']     = 2   # caso
controles_merged['PHENO'] = 1   # controle

# Combina em um único DataFrame e remove duplicatas
pheno = pd.concat([casos_merged, controles_merged], ignore_index=True)
pheno = pheno.drop_duplicates(subset=['sample_id'])

# Salva arquivo de fenótipos no formato PLINK (FID IID PHENO)
# Nota: FID = IID = sample_id (sem estrutura familiar)
pheno[['sample_id', 'sample_id', 'PHENO']].to_csv(
    'pheno.txt',
    sep='\t',
    index=False,
    header=['FID', 'IID', 'PHENO']
)

# Lê o arquivo salvo e padroniza o nome da coluna para merge posterior
IDs = pd.read_csv('pheno.txt', sep='\t')
IDs.rename(columns={'IID': 'participant_id'}, inplace=True)

print('Distribuição de fenótipos:')
print(IDs['PHENO'].value_counts())


# =============================================================================
# SEÇÃO 8 — INTEGRAÇÃO COM DADOS CLÍNICOS
# =============================================================================
# Sequência de merges para adicionar covariáveis ao DataFrame de IDs

# 1. Dados demográficos (idade, sexo, raça, etnia)
IDs_1 = pd.merge(IDs, demo, on='participant_id')

# 2. UPDRS III — escore motor mais recente
IDs_2 = pd.merge(IDs_1, III_recente, on='participant_id')

# 3. UPDRS IV — escore de complicações motoras mais recente
IDs_merged = pd.merge(IDs_2, IV_recente, on='participant_id')

# 4. Histórico médico — diagnóstico, idade ao diagnóstico e uso de levodopa
IDs_med = pd.merge(
    IDs_merged,
    med[['participant_id', 'most_recent_diagnosis', 'age_at_diagnosis', 'on_levodopa']],
    on='participant_id'
)

# Seleciona e renomeia apenas as colunas de interesse
colunas = [
    'FID', 'participant_id', 'PHENO',
    'age_at_baseline', 'sex', 'ethnicity', 'race',
    'mds_updrs_part_iii_summary_score',
    'mds_updrs_part_iv_summary_score',
    'age_at_diagnosis', 'on_levodopa', 'most_recent_diagnosis'
]
IDs_final = IDs_med[colunas].copy()
IDs_final.rename(columns={
    'participant_id':               'IID',
    'age_at_baseline':              'age',
    'mds_updrs_part_iii_summary_score': 'part_iii_score',
    'mds_updrs_part_iv_summary_score':  'part_iv_score'
}, inplace=True)

# Codifica sexo numericamente (exigido pelo PLINK)
# 1 = masculino, 2 = feminino
IDs_final['sex'] = IDs_final['sex'].replace({'Male': 1, 'Female': 2})


# =============================================================================
# SEÇÃO 9 — FILTRAGEM POR CRITÉRIOS DE INCLUSÃO
# =============================================================================

# Critério 1: etnia não-hispânica e raça branca
# (minimiza confundimento por estratificação populacional)
IDs_final_a = IDs_final.loc[
    (IDs_final['ethnicity'] == 'Not Hispanic or Latino') &
    (IDs_final['race'] == 'White')
]

# Critério 2: uso de levodopa confirmado
# (garante que todos os participantes tiveram exposição ao tratamento
#  que está associado ao desenvolvimento de discinesia)
IDs_final_b = IDs_final_a.loc[IDs_final_a['on_levodopa'] == 'Yes']

print('Distribuição de fenótipos após filtros de inclusão:')
print(IDs_final_b['PHENO'].value_counts(dropna=False))


# =============================================================================
# SEÇÃO 10 — CONTROLE DE QUALIDADE DE IDADES
# =============================================================================
# Remove amostras com valores de idade biologicamente implausíveis

# Diagnóstico das distribuições de idade
print('\nIdade na baseline:')
print(IDs_final_b['age'].describe())
print('\nIdade ao diagnóstico:')
print(IDs_final_b['age_at_diagnosis'].describe())

# Define critérios de exclusão por idade
cond_age_low         = IDs_final_b['age'] < 30                                       # idade de baseline muito baixa para DP
cond_diag_after_age  = IDs_final_b['age_at_diagnosis'] > IDs_final_b['age']          # diagnóstico posterior à baseline (impossível)
cond_diag_low        = IDs_final_b['age_at_diagnosis'] < 30                           # diagnóstico de DP antes dos 30 anos (raro, provável erro)

print(f'\nAmostras com age < 30: {cond_age_low.sum()}')
print(f'Amostras com diagnóstico > baseline: {cond_diag_after_age.sum()}')
print(f'Amostras com diagnóstico < 30: {cond_diag_low.sum()}')

to_remove = cond_age_low | cond_diag_after_age | cond_diag_low
print(f'Total a remover: {to_remove.sum()}')

IDs_final_b_filtered = IDs_final_b[~to_remove].copy()
print(f'\nAmostras antes da filtragem de idade: {len(IDs_final_b)}')
print(f'Amostras após a filtragem de idade : {len(IDs_final_b_filtered)}')


# =============================================================================
# SEÇÃO 11 — PREPARAÇÃO FINAL DO DATAFRAME DE COVARIÁVEIS
# =============================================================================

# Remove colunas que já cumpriram sua função de filtragem
IDs_final_b_filtered = IDs_final_b_filtered.drop(
    columns=['ethnicity', 'race', 'on_levodopa', 'most_recent_diagnosis']
)

# Calcula duração da doença (anos desde o diagnóstico até a baseline)
IDs_final_b_filtered['disease_duration'] = (
    IDs_final_b_filtered['age'] - IDs_final_b_filtered['age_at_diagnosis']
)

# Verificação de valores faltantes antes de salvar
print('\nValores faltantes por coluna:')
print(IDs_final_b_filtered.isna().sum())

# Salva covariáveis (sem PCs ainda — serão adicionados após a PCA)
IDs_final_b_filtered.to_csv('covariables_restrict.txt', sep=' ', index=False)

print('\nArquivo covariables_restrict.txt salvo.')


# =============================================================================
# SEÇÃO 12 — UPLOAD DO ARQUIVO DE COVARIÁVEIS PARA O GCS
# =============================================================================

%%bash

WORK_DIR="/home/jupyter/NOS1 - dyskinesia/edit"
cd "$WORK_DIR"

gsutil cp "covariables_restrict.txt" gs://fc-secure-5dd0ff90-839e-485e-9021-a5a7ad3060ec


# =============================================================================
# SEÇÃO 13 — CÁLCULO DE COMPONENTES PRINCIPAIS (PCA)
# =============================================================================
# A PCA é usada para capturar a estrutura de ancestralidade genômica
# e será incluída como covariável na análise de associação para
# controlar a estratificação populacional.

# Download do arquivo genômico mesclado (todos os cromossomos)
for ext in ['pgen', 'psam', 'pvar']:
    shell_do(f'gsutil -u {BILLING_PROJECT_ID} -m cp {PROJECT_BUCKET}/merged_genome.{ext} {WORK_DIR}')

# Etapa 1: LD pruning — remove variantes em desequilíbrio de ligação
# Parâmetros: janela de 50 SNPs, passo de 5, r² < 0.2
# --geno 0.05  → exclui variantes com >5% de dados faltantes
# --mind 0.05  → exclui amostras com >5% de dados faltantes
# --max-alleles 2 → mantém apenas variantes bialélicas
!plink2 --pfile data/merged_genome \
  --set-all-var-ids @:#:$r:$a \
  --geno 0.05 \
  --mind 0.05 \
  --max-alleles 2 \
  --indep-pairwise 50 5 0.2 \
  --out pca_prune

# Etapa 2: Cálculo das 10 primeiras componentes principais
!plink2 --pfile data/merged_genome \
  --set-all-var-ids @:#:$r:$a \
  --extract pca_prune.prune.in \
  --pca 10 \
  --out merged_genome_PCA


# =============================================================================
# SEÇÃO 14 — REMOÇÃO DE OUTLIERS DE PCA (critério ±4 DP)
# =============================================================================
# Amostras com PCs muito distantes da média indicam ancestralidade diferente
# da maioria da amostra. São removidas para evitar confundimento.

# Lê os valores de PCs e calcula média e desvio padrão de cada PC
fileIn   = 'merged_genome_PCA.eigenvec'
dictPCA  = {}

with open(fileIn) as f:
    f.readline()  # pula cabeçalho
    for line in f:
        cols = line.strip().split()
        for i in range(2, len(cols)):
            PC = f'PC{i-1}'
            dictPCA.setdefault(PC, []).append(float(cols[i]))

sumStat = {
    PC: {'mean': np.mean(vals), 'sd': np.std(vals)}
    for PC, vals in dictPCA.items()
}

# Identifica outliers: qualquer amostra fora de ±4 DP em algum dos 10 PCs
toRemove = set()
with open(fileIn) as f:
    next(f)
    for line in f:
        cols      = line.strip().split()
        sample_id = cols[1]
        for i in range(2, 12):          # PC1 até PC10 (colunas 2–11)
            PC    = f'PC{i-1}'
            val   = float(cols[i])
            media = sumStat[PC]['mean']
            sd    = sumStat[PC]['sd']
            if val < media - 4 * sd or val > media + 4 * sd:
                toRemove.add(sample_id)

# Salva lista de outliers no formato esperado pelo PLINK (FID IID)
with open('toRemove.txt', 'w') as f:
    for sample in toRemove:
        f.write(f'0\t{sample}\n')

total   = len(dictPCA['PC1'])
removed = len(toRemove)
print(f'Total de amostras : {total}')
print(f'Outliers removidos: {removed}')
print(f'Amostras mantidas : {total - removed}')


# =============================================================================
# SEÇÃO 15 — FILTRAGEM DAS COVARIÁVEIS PELOS OUTLIERS DE PCA
# =============================================================================

cov_file    = 'data/covariables_restrict.txt'
remove_file = 'toRemove.txt'
out_file    = 'covariables_restrict_1.txt'

cov_restrict = pd.read_csv(cov_file, sep=r'\s+', dtype=str)
outliers     = pd.read_csv(remove_file, sep='\t', header=None, names=['FID', 'IID'], dtype=str)

cov_restrict_filtered = cov_restrict[~cov_restrict['IID'].isin(outliers['IID'])]
cov_restrict_filtered.to_csv(out_file, sep='\t', index=False)

print(f'Original : {len(cov_restrict)} amostras')
print(f'Removidas: {len(outliers)} amostras')
print(f'Final    : {len(cov_restrict_filtered)} amostras → {out_file}')

print('\nDistribuição de fenótipos após remoção de outliers:')
print(cov_restrict_filtered['PHENO'].value_counts())


# =============================================================================
# SEÇÃO 16 — ADIÇÃO DOS COMPONENTES PRINCIPAIS AO ARQUIVO DE COVARIÁVEIS
# =============================================================================

# Lê os PCs calculados
pcs = pd.read_csv('merged_genome_PCA.eigenvec', sep='\t')
pcs = pcs.rename(columns={'#FID': 'FID'})

# Mantém apenas FID, IID e os 10 primeiros PCs
pc_cols = ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)]
pcs     = pcs[pc_cols]

# Merge: adiciona os PCs às covariáveis já filtradas
cov_final = cov_restrict_filtered.merge(pcs, on=['FID', 'IID'], how='left')

# Remove amostras sem PCs (não deveriam existir após as etapas anteriores)
cov_final = cov_final.dropna(subset=[f'PC{i}' for i in range(1, 11)])

# Garante que colunas numéricas estão com tipo correto
cols_numericas = ['age', 'age_at_diagnosis', 'disease_duration']
cov_final[cols_numericas] = cov_final[cols_numericas].apply(pd.to_numeric, errors='coerce')

# Verificação final: valores negativos ou implausíveis
for col in cols_numericas:
    n_neg  = (cov_final[col] < 0).sum()
    n_high = (cov_final[col] > 110).sum()
    print(f'{col}: negativos={n_neg}, >110={n_high}')

# Salva arquivo final
cov_final.to_csv(
    'covariables_restrict_final.txt',
    sep='\t',
    index=False,
    float_format='%.6f'
)
print('\nArquivo covariables_restrict_final.txt salvo.')


# =============================================================================
# SEÇÃO 17 — VISUALIZAÇÕES DE CONTROLE DE QUALIDADE
# =============================================================================

# ── Scree plot: variância explicada por cada PC ───────────────────────────────
eigenval = pd.read_csv('merged_genome_PCA.eigenval', header=None, names=['Eigenvalue'])
prop_var = eigenval['Eigenvalue'] / eigenval['Eigenvalue'].sum() * 100

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(prop_var) + 1), prop_var, marker='o')
plt.xlabel('Componente principal')
plt.ylabel('Variância explicada (%)')
plt.title('Scree plot — PCA genômica')
plt.tight_layout()
plt.show()

print('\nVariância explicada por PC (%):')
print(prop_var.round(2).to_string())


# ── PC1 vs PC2 colorido por sexo ─────────────────────────────────────────────
df_pca = pd.read_csv('covariables_restrict_final.txt', sep='\t')

plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    df_pca['PC1'], df_pca['PC2'],
    c=df_pca['sex'], cmap='coolwarm', alpha=0.7
)
plt.legend(*scatter.legend_elements(), title='Sexo (1=M, 2=F)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA — PC1 vs PC2')
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# SEÇÃO 18 — UPLOAD DOS ARQUIVOS FINAIS PARA O GCS
# =============================================================================

%%bash

WORK_DIR="/home/jupyter/NOS1 - dyskinesia/edit"
cd "$WORK_DIR"

# Covariáveis finais
gsutil cp "covariables_restrict_final.txt" gs://fc-secure-5dd0ff90-839e-485e-9021-a5a7ad3060ec

# Arquivo de fenótipos
gsutil cp "pheno.txt" gs://fc-secure-5dd0ff90-839e-485e-9021-a5a7ad3060ec

echo "Upload concluído."
