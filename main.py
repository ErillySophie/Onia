# importando as bibliotecas necessárias 
# usando pandas para usar os dataframes e scikit-learn para usar as suas funções de classificação e métrica F1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Para carregar os arquivos 
arquivo_treino = pd.read_csv("treino.csv")
arquivo_teste = pd.read_csv("teste.csv")

# Para treinar o modelo vamos dividir colunas nas variaveis x e y
X = arquivo_treino.drop(columns=["id", "target"])
y = arquivo_treino["target"]

# Removendo a coluna ID do arquivo de teste
X_test = arquivo_teste.drop(columns=["id"])

# Preparando as variaveis do treino e da validação
X_treino, X_val, y_treino, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Treino o modelo usando o método random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_treino, y_treino)
RandomForestClassifier(random_state=42)

# Usando modelo para classificar o treino para validação
y_pred = model.predict(X_val)

# verificando 
f1 = f1_score(y_val, y_pred, average="weighted")
print("F1 Score:", f1)

# usando o modelo para classificar nossos planetas
y_test_pred = model.predict(X_test)


# Salvando o Resultado em um arquivo csv (tabela)
df_result = pd.DataFrame({"id": arquivo_teste["id"], "target": y_test_pred})
df_result.to_csv("resultados.csv", index=False)

# Salvando arquivo com coluna nomeada
df_result2 = pd.DataFrame({"id": arquivo_teste["id"], "target": y_test_pred}) 
df_result2["classificação"] = df_result2["target"].map({
    0: "Planeta Deserto",
    1: "Planeta Vulcânico",
    2: "Planeta Oceânico",
    3: "Planeta Florestal",
    4: "Planeta Gelado"

})
df_result2.to_csv("resultados_nomeado.csv", index=False)