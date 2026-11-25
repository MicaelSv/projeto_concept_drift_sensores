from pathlib import Path
import hashlib
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ORDEM = ["Estatico_B1", "Ultimo_lote", "Janela_3", "Cumulativo", "Ensemble_recencia"]
ROTULOS = {
    "Estatico_B1": "Estático B1",
    "Ultimo_lote": "Último lote",
    "Janela_3": "Janela de 3 lotes",
    "Cumulativo": "Cumulativo",
    "Ensemble_recencia": "Ensemble por recência",
}
CORES = {
    "Estatico_B1": "#6b7280",
    "Ultimo_lote": "#dc2626",
    "Janela_3": "#f59e0b",
    "Cumulativo": "#059669",
    "Ensemble_recencia": "#2563eb",
}
NOMES_CLASSIFICADORES = {"SVM_RBF": "SVM com kernel RBF", "RandomForest": "Random Forest"}


def localizar_raiz():
    raiz = Path.cwd()
    if raiz.name == "notebooks":
        raiz = raiz.parent
    if not (raiz / "data" / "consolidated").is_dir():
        candidatos = [p.parent.parent.parent for p in raiz.rglob("data/consolidated/atributos.npz")]
        raiz = candidatos[0] if candidatos else raiz
    if not (raiz / "data" / "consolidated").is_dir():
        raise FileNotFoundError("Raiz do projeto não encontrada.")
    return raiz


def calcular_sha256(caminho):
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def verificar_manifesto(raiz, caminho_manifesto):
    manifesto = raiz / caminho_manifesto
    for linha in manifesto.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        esperado, relativo = linha.split("  ", 1)
        arquivo = raiz / relativo
        if not arquivo.is_file() or calcular_sha256(arquivo) != esperado:
            raise ValueError(f"Falha de integridade: {relativo}")


def criar_modelo(nome, configuracao, seed):
    if nome == "SVM_RBF":
        parametros = configuracao["classificadores"][nome]
        return make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel="rbf", C=float(parametros["C"]), gamma=parametros["gamma"], cache_size=2000),
        )
    if nome == "RandomForest":
        parametros = configuracao["classificadores"][nome]
        return RandomForestClassifier(
            n_estimators=int(parametros["n_estimators"]),
            max_features=parametros["max_features"],
            bootstrap=bool(parametros["bootstrap"]),
            class_weight=parametros["class_weight"],
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(f"Classificador desconhecido: {nome}")


def calcular_metricas(real, previsto):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanceada = balanced_accuracy_score(real, previsto)
    return {
        "acuracia": accuracy_score(real, previsto),
        "macro_f1_seis_classes": f1_score(
            real, previsto, labels=[1, 2, 3, 4, 5, 6], average="macro", zero_division=0
        ),
        "acuracia_balanceada_classes_presentes": balanceada,
    }


def executar_repeticao(nome, seed, configuracao, x, y, lotes):
    inicio = time.perf_counter()
    previsoes_lote = {}
    for lote_treino in range(1, 10):
        modelo = criar_modelo(nome, configuracao, seed)
        modelo.fit(x[lotes == lote_treino], y[lotes == lote_treino])
        for lote_teste in range(lote_treino + 1, 11):
            previsoes_lote[(lote_treino, lote_teste)] = modelo.predict(x[lotes == lote_teste])

    decaimento = float(configuracao["ensemble_decaimento"])
    registros = []
    previsoes = {}
    for lote_teste in range(2, 11):
        mascara_teste = lotes == lote_teste
        real = y[mascara_teste]
        candidatos = {
            "Estatico_B1": (previsoes_lote[(1, lote_teste)], lotes == 1),
            "Ultimo_lote": (previsoes_lote[(lote_teste - 1, lote_teste)], lotes == lote_teste - 1),
        }

        mascara_janela = (lotes >= max(1, lote_teste - 3)) & (lotes < lote_teste)
        modelo_janela = criar_modelo(nome, configuracao, seed)
        modelo_janela.fit(x[mascara_janela], y[mascara_janela])
        candidatos["Janela_3"] = (modelo_janela.predict(x[mascara_teste]), mascara_janela)

        mascara_cumulativa = lotes < lote_teste
        modelo_cumulativo = criar_modelo(nome, configuracao, seed)
        modelo_cumulativo.fit(x[mascara_cumulativa], y[mascara_cumulativa])
        candidatos["Cumulativo"] = (modelo_cumulativo.predict(x[mascara_teste]), mascara_cumulativa)

        votos = np.zeros((int(mascara_teste.sum()), 6), dtype=float)
        for lote_modelo in range(1, lote_teste):
            peso = decaimento ** ((lote_teste - 1) - lote_modelo)
            previsto_lote = previsoes_lote[(lote_modelo, lote_teste)].astype(int)
            votos[np.arange(len(previsto_lote)), previsto_lote - 1] += peso
        candidatos["Ensemble_recencia"] = (votos.argmax(axis=1) + 1, mascara_cumulativa)

        for estrategia in ORDEM:
            previsto, mascara_treino = candidatos[estrategia]
            previsoes[(estrategia, lote_teste)] = previsto
            registros.append(
                {
                    "classificador": nome,
                    "seed": int(seed),
                    "lote_teste": lote_teste,
                    "estrategia": estrategia,
                    **calcular_metricas(real, previsto),
                    "n_treino": int(mascara_treino.sum()),
                    "n_teste": int(mascara_teste.sum()),
                }
            )
    duracao = time.perf_counter() - inicio
    print(f"{NOMES_CLASSIFICADORES[nome]} | seed {seed}: {duracao:.1f}s")
    return registros, previsoes


def desvio_amostral(valores):
    return float(valores.std(ddof=1)) if len(valores) > 1 else 0.0


def ajustar_holm(valores_p):
    valores_p = np.asarray(valores_p, dtype=float)
    ordem = np.argsort(valores_p)
    ajustados = np.zeros(len(valores_p), dtype=float)
    acumulado = 0.0
    for posicao, indice in enumerate(ordem):
        valor = min(1.0, (len(valores_p) - posicao) * valores_p[indice])
        acumulado = max(acumulado, valor)
        ajustados[indice] = acumulado
    return ajustados


def main():
    raiz = localizar_raiz()
    verificar_manifesto(raiz, Path("checksums/originais.sha256"))
    manifesto_local = Path("data/consolidated/manifesto_local.sha256")
    verificar_manifesto(
        raiz,
        manifesto_local if (raiz / manifesto_local).is_file() else Path("checksums/consolidado.sha256"),
    )
    configuracao = json.loads(
        (raiz / "config" / "protocolo_validacao_classificadores.json").read_text(encoding="utf-8")
    )
    x = load_npz(raiz / "data" / "consolidated" / "atributos.npz")
    y = np.load(raiz / "data" / "consolidated" / "classes.npy", allow_pickle=False)
    lotes = pd.read_csv(raiz / "data" / "consolidated" / "origem.csv")["lote"].to_numpy(dtype=int)
    assert x.shape == (13910, 128) and len(y) == len(lotes)

    planos = [("SVM_RBF", 0)] + [
        ("RandomForest", seed) for seed in configuracao["classificadores"]["RandomForest"]["seeds"]
    ]
    todos_registros = []
    todas_previsoes = {}
    for nome, seed in planos:
        registros, previsoes = executar_repeticao(nome, seed, configuracao, x, y, lotes)
        todos_registros.extend(registros)
        for chave, previsto in previsoes.items():
            todas_previsoes[(nome, seed, *chave)] = previsto

    resultados = pd.DataFrame(todos_registros)
    por_lote = (
        resultados.groupby(["classificador", "estrategia", "lote_teste"], as_index=False)
        .agg(
            macro_f1_media_seeds=("macro_f1_seis_classes", "mean"),
            macro_f1_desvio_seeds=("macro_f1_seis_classes", desvio_amostral),
            acuracia_media_seeds=("acuracia", "mean"),
            acuracia_balanceada_media_seeds=("acuracia_balanceada_classes_presentes", "mean"),
            n_seeds=("seed", "nunique"),
            n_teste=("n_teste", "first"),
        )
    )
    maximos = por_lote.groupby(["classificador", "lote_teste"])["macro_f1_media_seeds"].transform("max")
    por_lote["vitoria_ou_empate"] = np.isclose(por_lote["macro_f1_media_seeds"], maximos).astype(int)

    resumo = (
        por_lote.groupby(["classificador", "estrategia"], as_index=False)
        .agg(
            macro_f1_media_lotes=("macro_f1_media_seeds", "mean"),
            macro_f1_desvio_lotes=("macro_f1_media_seeds", lambda s: s.std(ddof=1)),
            macro_f1_pior_lote=("macro_f1_media_seeds", "min"),
            variabilidade_seeds_media=("macro_f1_desvio_seeds", "mean"),
            acuracia_media_lotes=("acuracia_media_seeds", "mean"),
            acuracia_balanceada_media_lotes=("acuracia_balanceada_media_seeds", "mean"),
            vitorias_ou_empates=("vitoria_ou_empate", "sum"),
        )
        .sort_values(["classificador", "macro_f1_media_lotes"], ascending=[True, False])
    )

    reais = np.concatenate([y[lotes == lote] for lote in range(2, 11)])
    linhas_classe = []
    for nome, seed in planos:
        for estrategia in ORDEM:
            previstos = np.concatenate(
                [todas_previsoes[(nome, seed, estrategia, lote)] for lote in range(2, 11)]
            )
            valores = f1_score(
                reais, previstos, labels=[1, 2, 3, 4, 5, 6], average=None, zero_division=0
            )
            for classe, valor in zip(range(1, 7), valores):
                linhas_classe.append(
                    {
                        "classificador": nome,
                        "seed": seed,
                        "estrategia": estrategia,
                        "classe": classe,
                        "f1_pool_amostras": valor,
                    }
                )
    f1_seeds = pd.DataFrame(linhas_classe)
    f1_por_classe = (
        f1_seeds.groupby(["classificador", "estrategia", "classe"], as_index=False)
        .agg(
            f1_media_seeds=("f1_pool_amostras", "mean"),
            f1_desvio_seeds=("f1_pool_amostras", desvio_amostral),
        )
    )

    linhas_testes = []
    for nome in NOMES_CLASSIFICADORES:
        tabela = por_lote[por_lote["classificador"] == nome].pivot(
            index="lote_teste", columns="estrategia", values="macro_f1_media_seeds"
        )
        inicio_grupo = len(linhas_testes)
        for referencia in [e for e in ORDEM if e != "Cumulativo"]:
            diferencas = tabela["Cumulativo"] - tabela[referencia]
            teste = wilcoxon(diferencas, alternative="greater", zero_method="wilcox", method="approx")
            linhas_testes.append(
                {
                    "classificador": nome,
                    "comparacao": f"Cumulativo > {referencia}",
                    "delta_macro_f1_medio": diferencas.mean(),
                    "vitorias": int((diferencas > 1e-12).sum()),
                    "empates": int(np.isclose(diferencas, 0).sum()),
                    "derrotas": int((diferencas < -1e-12).sum()),
                    "estatistica_wilcoxon": teste.statistic,
                    "p_unilateral": teste.pvalue,
                }
            )
        fim_grupo = len(linhas_testes)
        ajustados = ajustar_holm([linha["p_unilateral"] for linha in linhas_testes[inicio_grupo:fim_grupo]])
        for linha, ajustado in zip(linhas_testes[inicio_grupo:fim_grupo], ajustados):
            linha["p_holm"] = ajustado
    testes = pd.DataFrame(linhas_testes)

    fig, eixos = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for eixo, nome in zip(eixos, ["SVM_RBF", "RandomForest"]):
        dados_nome = por_lote[por_lote["classificador"] == nome]
        for estrategia in ORDEM:
            grupo = dados_nome[dados_nome["estrategia"] == estrategia].sort_values("lote_teste")
            x_lotes = grupo["lote_teste"].to_numpy(dtype=int)
            medias = grupo["macro_f1_media_seeds"].to_numpy(dtype=float)
            desvios = grupo["macro_f1_desvio_seeds"].to_numpy(dtype=float)
            eixo.plot(x_lotes, medias, marker="o", linewidth=2, label=ROTULOS[estrategia], color=CORES[estrategia])
            if np.any(desvios > 0):
                eixo.fill_between(x_lotes, medias - desvios, medias + desvios, color=CORES[estrategia], alpha=0.10)
        eixo.set_title(NOMES_CLASSIFICADORES[nome])
        eixo.set_xlabel("Lote de teste")
        eixo.set_xticks(range(2, 11))
        eixo.set_ylim(0.15, 0.95)
        eixo.grid(alpha=0.25)
    eixos[0].set_ylabel("Macro-F1 (6 classes)")
    alcas, rotulos = eixos[1].get_legend_handles_labels()
    fig.legend(
        alcas,
        rotulos,
        fontsize=8,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("Robustez das estratégias em dois classificadores")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    arquivos = {
        raiz / "results" / "resultados_validacao_classificadores.csv": resultados,
        raiz / "reports" / "tables" / "comparacao_classificadores_por_lote.csv": por_lote,
        raiz / "reports" / "tables" / "resumo_validacao_classificadores.csv": resumo,
        raiz / "reports" / "tables" / "f1_por_classe_validacao_classificadores.csv": f1_por_classe,
        raiz / "reports" / "tables" / "testes_pareados_validacao_classificadores.csv": testes,
    }
    for caminho, tabela in arquivos.items():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(caminho, index=False, float_format="%.6f", lineterminator="\n")
    figura = raiz / "reports" / "figures" / "validacao_classificadores_macro_f1.png"
    figura.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figura, dpi=220, bbox_inches="tight")
    plt.show()

    print("\nResumo:")
    print(resumo.to_string(index=False))
    print("\nArtefatos gravados.")


if __name__ == "__main__":
    main()
