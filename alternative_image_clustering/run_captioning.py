from caption import VQA
from caption import LLavaVQA


if __name__ == "__main__":
    # model = VQA()
    model = LLavaVQA()
    model.generate("cards", "v1")
    model.generate("fruit360", "v1")
    # model.generate("gtsrb", "v1")
    # model.generate("nrobjects", "v1")
    # model.generate("fruit360", "v1")
