from image_embed import LLavaImageEmbedder


if __name__ == "__main__":
    # model = VQA()
    model = LLavaImageEmbedder()
    model.generate("cards", "v1")
    model.generate("fruit360", "v1")
    model.generate("gtsrb", "v1")
    model.generate("nrobjects", "v1")
