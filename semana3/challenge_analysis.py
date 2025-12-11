
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
import os

# Set HF_HOME if needed, but default cache should work
# os.environ["HF_HOME"] = "..." 

print("Carregando modelo...")
model_name = "neuralmind/bert-base-portuguese-cased"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit(1)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()

# 1. Lista original + novas palavras
palavras_base = [
    # Componentes
    "Resistor", "Capacitor", "Diodo", "Transistor", "Chip",
    # Ferramentas/Processos
    "Solda", "Multímetro", "Alicate", "Reflow", "Estanho",
    # Aleatórios
    "Pizza", "Futebol", "Praia", "Música"
]

# Palavras do desafio
# "STM32F407" é o Part Number mencionado no notebook
palavras_desafio = ["STM32F407", "Banco", "Assento", "Dinheiro"]
todas_palavras = palavras_base + palavras_desafio

print("Gerando embeddings...")
vetores = []
for p in todas_palavras:
    print(f"Processando: {p}")
    vetores.append(get_embedding(p))

vetores = np.array(vetores)

# 2. PCA para 2D
print("Calculando PCA...")
pca = PCA(n_components=2)
vetores_2d = pca.fit_transform(vetores)

# 3. Plotar e Salvar
plt.figure(figsize=(10, 8))
plt.scatter(vetores_2d[:, 0], vetores_2d[:, 1], c='blue')

# Destacar os pontos do desafio
indices_desafio = [todas_palavras.index(p) for p in palavras_desafio]
plt.scatter(vetores_2d[indices_desafio, 0], vetores_2d[indices_desafio, 1], c='red', s=100, label='Desafio')

for i, palavra in enumerate(todas_palavras):
    plt.annotate(palavra, xy=(vetores_2d[i, 0], vetores_2d[i, 1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.title("Mapa Semântico: Componentes vs. Aleatório vs. Desafio")
plt.grid(True)
plt.legend()
plt.savefig('desafio_plot.png')
print("Gráfico salvo em 'desafio_plot.png'")

# 4. Análise Numérica (Similaridade de Cosseno)
# Lembrar: Similaridade = 1 - Distância. Distância pequena = Similaridade alta.
def calc_sim(v1, v2):
    return 1 - cosine(v1, v2)

# Analisando STM32F407
idx_stm = todas_palavras.index("STM32F407")
vec_stm = vetores[idx_stm]

# Centróide dos componentes para comparação
indices_comp = [todas_palavras.index(p) for p in ["Resistor", "Capacitor", "Diodo", "Transistor", "Chip"]]
centroide_comp = np.mean(vetores[indices_comp], axis=0)
sim_stm_comp = calc_sim(vec_stm, centroide_comp)

# Centróide dos aleatórios
indices_rand = [todas_palavras.index(p) for p in ["Pizza", "Futebol", "Praia", "Música"]]
centroide_rand = np.mean(vetores[indices_rand], axis=0)
sim_stm_rand = calc_sim(vec_stm, centroide_rand)

print("\n--- Análise 1: Part Number (STM32F407) ---")
print(f"Similaridade com Componentes (Centróide): {sim_stm_comp:.4f}")
print(f"Similaridade com Coisas Aleatórias (Centróide): {sim_stm_rand:.4f}")
if sim_stm_comp > sim_stm_rand:
    print("Conclusão: Caiu mais perto dos Componentes.")
else:
    print("Conclusão: Caiu mais isolado ou perto de Aleatórios.")

# Analisando Banco
idx_banco = todas_palavras.index("Banco")
idx_assento = todas_palavras.index("Assento")
idx_dinheiro = todas_palavras.index("Dinheiro")

sim_banco_assento = calc_sim(vetores[idx_banco], vetores[idx_assento])
sim_banco_dinheiro = calc_sim(vetores[idx_banco], vetores[idx_dinheiro])

print("\n--- Análise 2: A Palavra 'Banco' ---")
print(f"Similaridade Banco <-> Assento: {sim_banco_assento:.4f}")
print(f"Similaridade Banco <-> Dinheiro: {sim_banco_dinheiro:.4f}")

with open("analysis_results.txt", "w", encoding="utf-8") as f:
    f.write(f"--- Análise 1: Part Number (STM32F407) ---\n")
    f.write(f"Similaridade com Componentes (Centróide): {sim_stm_comp:.4f}\n")
    f.write(f"Similaridade com Coisas Aleatórias (Centróide): {sim_stm_rand:.4f}\n")
    if sim_stm_comp > sim_stm_rand:
        f.write("Conclusão: Caiu mais perto dos Componentes.\n")
    else:
        f.write("Conclusão: Caiu mais isolado ou perto de Aleatórios.\n")
    
    f.write(f"\n--- Análise 2: A Palavra 'Banco' ---\n")
    f.write(f"Similaridade Banco <-> Assento: {sim_banco_assento:.4f}\n")
    f.write(f"Similaridade Banco <-> Dinheiro: {sim_banco_dinheiro:.4f}\n")
    if sim_banco_assento > sim_banco_dinheiro:
        f.write("O modelo associou 'Banco' mais a 'Assento' (móvel).\n")
    else:
        f.write("O modelo associou 'Banco' mais a 'Dinheiro' (instituição financeira).\n")

if sim_banco_assento > sim_banco_dinheiro:
    print("O modelo associou 'Banco' mais a 'Assento' (móvel).")
else:
    print("O modelo associou 'Banco' mais a 'Dinheiro' (instituição financeira).")
