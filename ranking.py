#import pandas as pd

#quizas no lo hacemos nosotros

#iterable == vector de participantes 
#ranking ordinal mejorable a ranking 'denso' 
#def o_rank(puntuacions ):
#    """Ordinal  ranking"""
#    start = 1
#    yield from enumerate(puntuacions, start)  #yield: nuevo generador con esos parametros

#Implementing bubble sorter

def bubble_sort(punctuations, users):
    has_swapped = True

    num_of_iterations = 0

    while(has_swapped):
        has_swapped = False
        for i in range(len(punctuations) - num_of_iterations - 1):
            if punctuations[i] > punctuations[i+1]:
                # Swap
                punctuations[i], punctuations[i+1] = punctuations[i+1], punctuations[i]
                users[i], users[i+1] = users[i+1], users[i]
                has_swapped = True
        num_of_iterations += 1
    
    for j in range(len(punctuations)):
        pos[j] = j

