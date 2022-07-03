from DecoNeuralSRC.model import DecoNeuralModel
from DecoNeuralSRC.preprocessing import ASTTokenizer

def start_mode(mode, arch, asm_path, c_path, nb_files, train_eval_ratio, batch_size, model_size, model_save, asm_decompile = None, c_decompile = None):
    tokenizer = ASTTokenizer(arch)
    train_dataset = tokenizer.load_dataset(asm_path, c_path, is_train=True, nb_files=nb_files,train_eval_ratio=train_eval_ratio)
    dataset = tokenizer.preprocess_dataset(train_dataset)
    dataset = tokenizer.tokenize(dataset)
    dataset = dataset.shuffle(nb_files).batch(batch_size, drop_remainder=True)
    model = DecoNeuralModel(tokenizer, dataset, model_size, asm_path, c_path)
    epochs = 50
    if mode == "train":
        model.train(epochs)
        model.predict()
        model.save(model_save)
    elif mode == "predict":
        model.load(model_save)
        model.predict(asm_decompile, c_decompile, verbose=True)
    elif mode == "proceed":
        model.train(epochs, verbose=False, predict=False)
        model.save("save_proceed")

if __name__ == "__main__":
    asm_decompile = "rd_19959.s"
    c_decompile = "rd_19959.c"
    arch = "x86"
    asm_path = "./data_20_000/asm_src"
    c_path = "./data_20_000/c_src"
    nb_files = 20_000
    train_eval_ratio = .9
    batch_size = 8
    model_size = 64
    model_save = "save"
    mode = ["train","predict","proceed"]
    start_mode("predict",arch, asm_path, c_path, nb_files, train_eval_ratio, batch_size, model_size, model_save, asm_decompile, c_decompile)