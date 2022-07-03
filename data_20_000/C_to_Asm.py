import os

def Clang(c_src,asm_src):
    files = os.listdir(c_src)
    filenames = filter(lambda x: x.endswith('.c'), files)
    i = len(files)
    for c, filename in enumerate(filenames):
        print(c+1, "/", i)
        os.system(f'clang {c_src}/{filename} -O0 -S -o "{asm_src}/{filename.split(".")[0]}.s"')

if __name__ == "__main__":
    c_src = 'c_src'
    asm_src = 'asm_src'
    cwd = os.getcwd()
    fp = os.path.join(cwd, asm_src)
    if not os.path.exists(fp):
       os.mkdir(fp)
    Clang(c_src, asm_src)