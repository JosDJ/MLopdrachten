import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from scipy.sparse import csr_matrix


# ==== OPGAVE 1 ====
def plot_number(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

    plt.matshow(np.reshape(nrVector, (20, 20), order='F'))
    #plt.show()


# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.
    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m

    # YOUR CODE HERE
    rows = [i for i in range(m)]  # alle getallen 0-4999 de
    data = [1 for _ in range(m)]  # 5000 x een 1
    col = y[:, 0]  # alle waarden van y zo zodat we een vector krijgen
    width = np.max(y)  # breedte van return matrix
    y_vec = csr_matrix((data, (rows, col)), shape=(m, width + 1)).toarray()
    y_vec = y_vec[0:m, 1:]  # alle eerste 0'en eraf halen --> zorgt er wel voor dat alle 1->2 2->3 etc
    return y_vec


# ==== OPGAVE 2c ====
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    # a1 shape is 5000 401 --> theta shape is (401 25) (? volgens mij is de waarde andersom) dus theta transposen
    a1 = np.insert(X,0,1, axis=1)
    z2 = np.dot(a1,Theta1.T)

    a2 = sigmoid(z2)

    # theta weer transposen want a2 5000 26 en theta is 10 26
    a2 = np.insert(a2,0,1, axis=1)
    z3 = np.dot(a2,Theta2.T)

    # out is nu 5000x10
    out = sigmoid(z3)

    return out


# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix.
    m = X.shape[0]
    y_matr = get_y_matrix(y, m)
    p = predict_number(Theta1,Theta2,X)
    cost = np.sum((y_matr * np.log(p)) + ((1-y_matr) * np.log(1-p)))
    J = -cost/m

    return J



# ==== OPGAVE 3a ====
def sigmoid_gradient(z):
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    out = sigmoid(z) - np.power(sigmoid(z),2)
    return out[0]



# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y):
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = X.shape[0] #???

    a1 = np.insert(X, 0, 1, axis=1)
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    #Delta3 moet iets van 10,26 zijn
    #Delta2 moet iets van 25,401 zijn
    y_matrix = get_y_matrix(y,m)
    d3 = a3 - y_matrix #10,5000
    d2 = Theta2.T.dot(d3.T) # 26,5000
    d2 = np.delete(d2,0,0)
    d2 = d2.T * sigmoid_gradient(z2)
    Delta2 += np.dot(a1.T, d2).T
    Delta3 += np.dot(a2.T,d3).T

    # for i in range(m):
    #     # YOUR CODE HERE
    #     pass

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad
