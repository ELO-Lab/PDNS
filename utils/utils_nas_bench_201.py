op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
available_ops = [0, 1, 2, 3, 4]

def convert_arch_var_to_str(arch_var):
        arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
            op_names[arch_var[0]],
            op_names[arch_var[1]],
            op_names[arch_var[2]],
            op_names[arch_var[3]],
            op_names[arch_var[4]],
            op_names[arch_var[5]],
            )
        return arch_str

def convert_str_to_op_indices(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split('+')

    def get_op(x):
        return x.split('~')[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return tuple(enc)

def convert_arch_var_to_arch_var_str(arch_var):
    # Convert each item in the list to a string and concatenate them
    arch_var_str = ''.join(map(str, arch_var))
    return arch_var_str