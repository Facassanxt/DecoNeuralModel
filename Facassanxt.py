from DecoNeuralSRC.model import DecoNeuralModel
from DecoNeuralSRC.preprocessing import ASTTokenizer
import os

def start_mode(mode, arch, asm_path, c_path, nb_files, train_eval_ratio, batch_size, model_size, model_save):
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
        while True:
            c_decompile = input('Введите название С файла: \n')
            if len(c_decompile) < 1:
                break
            c_decompile = c_decompile + ".c"
            #os.system(f'clang {c_decompile} -O0 -S')
            asm_decompile = c_decompile.split(".")[0] + ".s"
            try:
                model.predict(asm_decompile, c_decompile, verbose=True)
            except Exception as e:
                print(e)
    elif mode == "proceed":
        model.train(epochs, verbose=False, predict=False)
        model.save("save_proceed")
    elif mode == "test":
        model.load(model_save)
        bleu_scores = []
        for i in range(18_000, 20_000):
            print("="*10,i,"="*10)
            asm_file = os.path.join(asm_path, f"rd_{i}.s")
            c_file = os.path.join(c_path, f"rd_{i}.c")
            model.predict(asm_file, c_file, verbose=False)
            bleu_scores.append(model.bleu_score)
        mean = sum(bleu_scores) / len(bleu_scores)
        print('Среднее значение - {:%}'.format(mean))

if __name__ == "__main__":
    arch = "x86"
    asm_path = "./data_20_000/asm_src"
    c_path = "./data_20_000/c_src"
    nb_files = 20_000
    train_eval_ratio = .9
    batch_size = 8
    model_size = 64
    model_save = "save"
    #mode = ["train","predict","proceed","test"]
    start_mode("predict",arch, asm_path, c_path, nb_files, train_eval_ratio, batch_size, model_size, model_save)


