from pickle import load as pload
from do_embed import normalize, client
from numpy import dot

with open("embeddings.pkl", "rb") as f:
    embeddings: dict[str, list[float]] = pload(f)


def similarity(a, b) -> float:
    return 1 - dot(a, b)


while True:
    try:
        user_in = input("Enter a sentence: ")
    except KeyboardInterrupt:
        break
    if user_in == "!exit":
        break
    user_in_embed = normalize(
        client.embeddings.create(
            input=user_in, model="text-embedding-3-large", dimensions=3072
        )
        .data[0]
        .embedding
    )
    best_match = sorted(
        embeddings.keys(), key=lambda k: similarity(user_in_embed, embeddings[k])
    )
    print(f"Best matches: {'\n'.join(best_match[:3])}")
