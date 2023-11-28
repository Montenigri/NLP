#
#Questo script serve a replicare il comportamento di eliza, sarà possibile selezionare se leggere il contenuto tramite regex
# o tramite l'uso di funzioni integrate in python dove viene definito un array di parole da confrontare con l'input
#

import re

#Array di parole che includono sentimenti negativi
neg = ["trist","cattiv","pessim","depress","mort"]

def regexParsing(input):
    #trova tutte le occorrenze che contengono la parole "trist","cattiv","pessim","depress" con uno spazio prima ed un qualsiasi carattere dopo
    #e restituisce l'occorrenza
    #Se non trova occorrenze ritorna stringa vuota

    occorrenze= re.search(r"\b(trist\w+|cattiv\w+|pessim\w+|depress\w+|mort\w+)\b",input+" ")
    if not occorrenze:
        return ""
    return occorrenze 
    


def risposta(Input):
    #Controlla se all'interno dell'input è presente una delle stringhe presenti nell'array neg precededute da uno spazio e seguite da un carattere qualsiasi, quindi ritorna la parola trovata compresa di spazio precedente e caratteri seguenti fino al primo spazio
    parsedInput = ""
    preprocess = Input.split()
    for i in neg:
        for j in preprocess:
            if j.startswith(i):
                return j
        
    return parsedInput

def eliza():
    #preso un input permette di selezioare se leggere il contenuto tramite regex o tramite l'uso di funzioni integrate in python
    choose = input("0 per regex, 1 per python: ")
    if choose == "0":
        print("Ciao, sono Eliza regex: ")
        while(True): 
            #leggo l'input e lo trasformo in lowercase
            inp = input().lower()
            parsedInput = regexParsing(inp)
            if parsedInput == "":
                print("Pensi che parlarne ti aiuterebbe?")
            else:
                print(f"Cosa ti fa sentire {parsedInput[0]}?")

    else:
        print("Ciao, sono Eliza python: ")
        while(True):
            inp = input().lower()
            #Se la risposta è stringa vuota stampa una stringa predefinita
            if risposta(inp) == "":
                print("Pensi che parlarne ti aiuterebbe?")
            else:
                print(f"Cosa ti fa sentire {risposta(inp)}?")

if __name__ == "__main__":
    eliza()