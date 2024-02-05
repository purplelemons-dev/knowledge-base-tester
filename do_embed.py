from openai import OpenAI
from numpy.typing import NDArray, ArrayLike
from numpy import float64 as f64
from numpy.linalg import norm
from pickle import dump as pdump

client = OpenAI()


def normalize(embed: ArrayLike) -> NDArray[f64]:
    return embed / norm(embed)


if __name__ == "__main__":
    with open("corpus.txt", "r") as f:
        corpus = f.read().split("\n")

    embeddings: dict[str, list[float]] = {}
    for i in range(0, len(corpus), 32):
        section = corpus[i : i + 32]
        for text, embed in zip(
            section,
            client.embeddings.create(
                input=section, model="text-embedding-3-large", dimensions=3072
            ).data,
        ):
            embeddings[text] = normalize(embed.embedding).tolist()

    # with open("embeddings.json", "w") as f:
    #    dump(embeddings, f, indent=2)

    with open("embeddings.pkl", "wb") as f:
        pdump(embeddings, f)
