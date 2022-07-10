import os
from typing import Generator
import tensorflow as tf
from DecoNeuralSRC.ast.ast_builder import extract_ast

class Tokenizer(object):
    def __init__(self):
        pass

    def load_dataset(self, *args, **kwargs) -> Generator:
        raise NotImplementedError()

    def preprocess_dataset(self, *args, **kwargs) -> Generator:
        raise NotImplementedError()

    def postprocess_prediction(
        self,
        c_code: str,
    ) -> str:
        def postprocess_c(c_code: str) -> str:
            tokens = ['=', '(', ')', ';', ',', '>', '{', '}']
            for tok in tokens:
                c_code = c_code.replace(f' {tok}', tok)
                c_code = c_code.replace(f'{tok} ', tok)
            return c_code
        return postprocess_c(c_code)

    def tokenize(
        self,
        dataset: Generator,
    ) -> tf.data.Dataset:
        raw_data_asm, raw_data_c = list(zip(*dataset))
        raw_data_asm, raw_data_c = list(raw_data_asm), list(raw_data_c)
        raw_data_c_in = ['<start> ' + data for data in raw_data_c]
        raw_data_c_out = [data + ' <end>' for data in raw_data_c]
        self.asm_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\t\n')
        self.asm_tokenizer.fit_on_texts(raw_data_asm)
        self.data_asm = self.asm_tokenizer.texts_to_sequences(raw_data_asm)
        self.data_asm = tf.keras.preprocessing.sequence.pad_sequences(self.data_asm, padding='post')
        self.c_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.c_tokenizer.fit_on_texts(raw_data_c_in)
        self.c_tokenizer.fit_on_texts(raw_data_c_out)
        self.data_c_in = self.c_tokenizer.texts_to_sequences(raw_data_c_in)
        self.data_c_in = tf.keras.preprocessing.sequence.pad_sequences(self.data_c_in, padding='post')
        self.data_c_out = self.c_tokenizer.texts_to_sequences(raw_data_c_out)
        self.data_c_out = tf.keras.preprocessing.sequence.pad_sequences(self.data_c_out, padding='post')
        ds = tf.data.Dataset.from_tensor_slices((self.data_asm, self.data_c_in, self.data_c_out))
        self.max_length = max(len(self.data_asm[0]), len(self.data_c_in[0]))
        return ds

class ASTTokenizer(Tokenizer):
    def __init__(
        self,
        arch: str,
    ):
        super(ASTTokenizer, self).__init__()
        self.arch = arch

    def load_dataset(
        self,
        x_dir: str,
        y_dir: str,
        is_train: bool,
        nb_files: int,
        train_eval_ratio: str=0.9,
    ) -> Generator:
        self.nb_files = nb_files
        self.train_eval_ratio = train_eval_ratio
        eval_threshold = nb_files * train_eval_ratio
        for i in range(nb_files):
            if (is_train and i < eval_threshold) or (not is_train and i >= eval_threshold):
                x_path = os.path.join(x_dir, f'rd_{i}.s')
                y_path = os.path.join(y_dir, f'rd_{i}.c')
                c_code = open(y_path).read()
                ast_str = extract_ast(x_path, self.arch)
                yield ast_str, c_code

    def preprocess_dataset(
        self,
        dataset: Generator,
    ) -> Generator:
        def preprocess_c(c_code: str) -> str:
            tokens = ['=', '(', ')', ';', ',', '{', '}']
            for tok in tokens:
                c_code = c_code.replace(tok, f' {tok} ')
            return c_code
        for (x, y) in dataset:
            y = preprocess_c(y)
            yield (x, y)