
def categorical(x_ri, y_si):
    return 0 if x_ri == y_si else 1

def interval(x_ri, y_si):
    return (x_ri - y_si)**2