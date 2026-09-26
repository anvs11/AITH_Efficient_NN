import numpy as np

def flops(S, B):
    return (17712 * S ** 2 + 313344) * B

def memory(S, B):
    return 4 * (26 * B * S ** 2 + 868 * B + 1039968)

def bytes_moved(S, B):
    return 4 * (91 * B * S ** 2 + 2148 * B + 1039968)

# ввел упрощения для формул latency и energy в ходе калибровки
def latency(S, B, theta):
    # return theta[0] * flops(S, B) + theta[1] * bytes_moved(S, B) + theta[2]
    return theta[0] * bytes_moved(S, B) + theta[1]

def energy(S, B, alpha):
    # return alpha[0] * flops(S, B) + alpha[1] * bytes_moved(S, B) + alpha[2]
    return alpha[0] * bytes_moved(S, B) + alpha[1]
