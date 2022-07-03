from DecoNeuralSRC.ast.x86_ast import Graphs_build_x86

def extract_ast(path: str, arch: str):
    assert arch in ["x86"], "Invalid architecture 'x86'"
    asm_nodes, asm_edges = Graphs_build_x86(path, '')
    ast_str = ""
    for edge in asm_edges:
        ast_str += ' '.join(list(map(lambda x: asm_nodes[x], edge))) + ' '

    if arch == "x86":
        ast = ""
        splitted = ast_str.split()
        for word in splitted:
            if word != "ins_nop":
                ast += word + " "
        ast_str = ast
    
    return ast_str
