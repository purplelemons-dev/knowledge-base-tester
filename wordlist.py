from do_embed import client, normalize
from pickle import dump as pdump

if __name__ == "__main__":
    with open("corpus.txt", "r", encoding="utf-8") as f:
        corpus_list = "".join(i for i in f.read() if ord(i) < 128).split()

    embeddings: dict[str, list[float]] = {}
    for idx in range(0, len(corpus_list), 1024):
        section = corpus_list[idx : idx + 1024]
        for text, embed in zip(
            section,
            client.embeddings.create(
                input=section, model="text-embedding-3-large", dimensions=3072
            ).data,
        ):
            embeddings[text] = normalize(embed.embedding).tolist()

    with open("per_word_embeddings.pkl", "wb") as f:
        pdump(embeddings, f)
