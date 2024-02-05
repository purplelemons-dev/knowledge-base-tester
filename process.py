import os

corpus_text: list[str] = []
for file in os.listdir("corpus_folder"):
    with open(os.path.join("corpus_folder", file), "r") as f:
        text = (i for i in (ii.strip() for ii in f.read().split("\n")) if i)
        processed: list[str] = []
        for line in text:
            if ". " in line:
                processed.extend(f"{i}." for i in line.split(". ") if i)
            else:
                processed.append(line)
        corpus_text.extend(processed)

with open("corpus.txt", "w") as f:
    f.write(
        "\n".join(
            i.lower()
            .replace("�", "")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", "-")
            .replace("–", "-")
            .replace("..", ".")
            .replace("…", "...")
            for i in corpus_text
        )
    )
