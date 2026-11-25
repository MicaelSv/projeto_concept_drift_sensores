from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, vstack
from sklearn.datasets import load_svmlight_file


ARQUIVOS_CONSOLIDADOS = ("atributos.npz", "classes.npy", "origem.csv", "metadados.json")


def calcular_sha256(caminho):
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def verificar_manifesto(raiz, manifesto):
    for linha in manifesto.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        esperado, relativo = linha.split("  ", 1)
        arquivo = raiz / relativo
        if not arquivo.is_file() or calcular_sha256(arquivo) != esperado:
            raise ValueError(f"Falha de integridade: {relativo}")


def localizar_raiz():
    raiz = Path(__file__).resolve().parent.parent
    if not (raiz / "checksums" / "originais.sha256").is_file():
        raise FileNotFoundError("A raiz do projeto não foi localizada.")
    return raiz


def consolidar(raiz, sobrescrever=False):
    pasta_saida = raiz / "data" / "consolidated"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    saidas = [pasta_saida / nome for nome in ARQUIVOS_CONSOLIDADOS]
    manifesto_local = pasta_saida / "manifesto_local.sha256"

    if any(caminho.exists() for caminho in saidas) and not sobrescrever:
        raise FileExistsError(
            "A base consolidada já existe. Use --force para recriá-la depois de conferir o caminho."
        )

    matrizes, rotulos, origem, resumo_lotes = [], [], [], []
    id_global = 0
    for lote in range(1, 11):
        caminho = raiz / "data" / "raw" / f"batch{lote}.dat"
        x, y = load_svmlight_file(caminho, n_features=128, dtype=np.float64)
        if not np.isfinite(x.data).all() or not np.isfinite(y).all():
            raise ValueError(f"Valor não finito no lote {lote}.")
        if not np.all(y == np.floor(y)) or not set(y).issubset(set(range(1, 7))):
            raise ValueError(f"Rótulo inválido no lote {lote}.")

        y = y.astype(np.int16)
        matrizes.append(x.tocsr())
        rotulos.append(y)
        for linha_local in range(len(y)):
            origem.append(
                {
                    "id_global": id_global,
                    "lote": lote,
                    "linha_origem_0": linha_local,
                    "arquivo_origem": f"batch{lote}.dat",
                }
            )
            id_global += 1
        resumo_lotes.append(
            {
                "lote": lote,
                "registros": len(y),
                "classes_presentes": np.unique(y).tolist(),
            }
        )
        print(f"Lote {lote}: {len(y)} registros")

    x_final = vstack(matrizes, format="csr")
    y_final = np.concatenate(rotulos)
    if x_final.shape != (13910, 128) or len(y_final) != 13910:
        raise ValueError(f"Dimensões inesperadas: X={x_final.shape}, y={y_final.shape}.")

    save_npz(saidas[0], x_final, compressed=True)
    np.save(saidas[1], y_final, allow_pickle=False)
    pd.DataFrame(origem).to_csv(saidas[2], index=False, lineterminator="\n")
    info = {
        "gerado_utc": datetime.now(timezone.utc).isoformat(),
        "registros": len(y_final),
        "atributos": 128,
        "classes": [1, 2, 3, 4, 5, 6],
        "lotes": resumo_lotes,
        "ordem": "B1 até B10, sem timestamps individuais",
        "sha256_manifesto_originais": calcular_sha256(raiz / "checksums" / "originais.sha256"),
        "versoes": {
            biblioteca: version(biblioteca)
            for biblioteca in ("numpy", "pandas", "scipy", "scikit-learn")
        },
    }
    saidas[3].write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifesto_local.write_text(
        "".join(
            f"{calcular_sha256(caminho)}  {caminho.relative_to(raiz).as_posix()}\n"
            for caminho in saidas
        ),
        encoding="utf-8",
    )
    print(f"Base consolidada criada em: {pasta_saida}")


def main():
    parser = argparse.ArgumentParser(description="Consolida os dez lotes do dataset da UCI.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="sobrescreve os quatro arquivos consolidados existentes",
    )
    argumentos = parser.parse_args()
    raiz = localizar_raiz()
    verificar_manifesto(raiz, raiz / "checksums" / "originais.sha256")
    consolidar(raiz, sobrescrever=argumentos.force)


if __name__ == "__main__":
    main()
