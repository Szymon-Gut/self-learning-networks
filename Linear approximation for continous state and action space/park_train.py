import numpy as np
import parking_model as pm
from numba import njit


GLOBAL_VARS = pm.GlobalVar()

PREDEFINED_ACTIONS = [
    [angle, speed]
    for angle in np.arange(
        -GLOBAL_VARS.wheel_turn_angle_max,
        GLOBAL_VARS.wheel_turn_angle_max + np.pi / 8,
        np.pi / 8,
    )
    for speed in np.arange(-GLOBAL_VARS.Vmod, GLOBAL_VARS.Vmod + 1, 1)
    if not speed == 0
]
PREDEFINED_ACTIONS.append([0, 0])
NUM_OF_ACTIONS = len(PREDEFINED_ACTIONS)


iht_size = 8192 
num_tilings = 8  
tile_size = [
    0.1,
    0.1,
    np.pi / 8,
    np.pi / 8,
    0.5
] 
offsets = [
    (i / num_tilings) * np.array(tile_size) for i in range(num_tilings)
] 

@njit
def tile_hash(indices, iht_size):
    return sum([index * (i + 1) for i, index in enumerate(indices)]) % iht_size

def get_tiles(state, action, iht_size=iht_size):
    tile_vector = np.zeros(iht_size)
    combined_state_action = np.concatenate(
        (state, action), axis=0
    )
    for offset in offsets:
        tile_index = np.floor((combined_state_action + offset) / tile_size).astype(int)
        index = tile_hash(tile_index, iht_size)
        tile_vector[index] = 1

    return tile_vector


class LinearApproximation:
    def __init__(self, num_features):
        self.weights = np.zeros((num_features))

    def predict(self, features):
        return np.dot(self.weights, features)

    def update(self, features, delta, alpha):
        self.weights += alpha * delta * features


def nagroda_za_krok(param_fiz, stan, czy_kolizja, czy_zatrzymanie):
    x, y, alfa = stan
    odl_xy_kw = np.sqrt(x**2 + y**2)
    if param_fiz.if_side_parking_place:
        if np.abs(alfa) > np.pi / 2:
            alfa_zred = np.pi - np.abs(alfa)
        else:
            alfa_zred = np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)

    alfa_zred = alfa_zred / (odl_xy_kw + 0.5)

    ocena_odl = 1 / (odl_xy_kw + 0.5) - 1
    ocena_alfa = alfa_zred - 0.5

    if czy_kolizja:
        wartosc = -1
    elif czy_zatrzymanie:
        wartosc = min(ocena_odl, ocena_alfa)
    else:
        wartosc = 0

    return wartosc


def choose_action(stan, model, param_fiz = GLOBAL_VARS):
    Q = np.array([model.predict(get_tiles(stan, action)) for action in PREDEFINED_ACTIONS])
    best_action_idx = np.argmax(Q)
    best_action = PREDEFINED_ACTIONS[best_action_idx]
    angle, velocity = best_action
    czy_zatrzymanie = 0
    if velocity == 0:
        czy_zatrzymanie = 1
    return angle, velocity, czy_zatrzymanie


def epsilon_greedy(state, model, epsilon):
    czy_zatrzymanie = 0
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(NUM_OF_ACTIONS)
        action = PREDEFINED_ACTIONS[action_idx]
        angle, velocity = action
        if velocity == 0:
            czy_zatrzymanie = 1
    else:
        angle, velocity, czy_zatrzymanie = choose_action(state, model)
    return angle, velocity, czy_zatrzymanie


def estimate_Q_values(stan, action, model):
    encoded = get_tiles(stan, action)
    return model.predict(encoded)

def park_train():
    liczba_epizodow = 5000
    alfa, epsilon = 0.002, 1.
    epsilon_decay = 0.9995  
    gamma = 0.99
    stany_poczatkowe_1 = np.array(
        [
            [9.1, 4.6, 0], 
            [6.3, 5.06, 0], 
            [9.6, 3.15, 0], 
            [7.3, 5.75, 0], 
            [10.1, 6.21, 0]
            ]
    )

    model = LinearApproximation(iht_size)
    for epizod in range(liczba_epizodow):
        epsilon = max(0.1, epsilon * epsilon_decay)
        stan = stany_poczatkowe_1[epizod % len(stany_poczatkowe_1), :]
        # stan = stany_poczatkowe_1
        krok, czy_kolizja, czy_zatrzymanie = 0, False, False

        while not czy_zatrzymanie:
            krok += 1
            kat, V, czy_zatrzymanie = epsilon_greedy(stan, model, epsilon)
            nowystan, srodek_rotacji, czy_kolizja = pm.model_of_car(GLOBAL_VARS, stan, kat, V)

            czy_zatrzymanie = czy_kolizja or (krok >= GLOBAL_VARS.max_number_of_steps)

            R = nagroda_za_krok(GLOBAL_VARS, nowystan, czy_kolizja, czy_zatrzymanie)

            biezaca_akcja = np.array([kat, V])
            Q_stanu_obecnego = estimate_Q_values(stan, biezaca_akcja, model)

            next_kat, next_v, czy_zatrzymanie_next = choose_action(nowystan, model, GLOBAL_VARS)
            nowa_akcja = np.array([next_kat, next_v])

            Q_stanu_kolejnego = estimate_Q_values(nowystan, nowa_akcja, model)

            delta = R + gamma * Q_stanu_kolejnego - Q_stanu_obecnego
            
            features = get_tiles(stan, biezaca_akcja)

            model.update(features, delta, alfa)
            stan = nowystan
        if epizod % 100 == 0:
            print("epizod %d\n" % epizod)
            park_test(GLOBAL_VARS, stany_poczatkowe_1, model, "historia_park.txt")
            
    print("Test dla losowych stanów początkowych:")
    stany_pocz_losowe = pm.random_initial_states(GLOBAL_VARS, 20)
    park_test(GLOBAL_VARS, stany_pocz_losowe, model, "historia_park_los.txt")


def park_test(param_fiz, stany_poczatkowe, model, nazwa_pliku):
    pm.park_save("param.txt", param_fiz)
    phist = open(nazwa_pliku, "w")
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape
    sr_ocena_koncowa = 0
    sr_liczba_krokow = 0
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod % liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan zgodnie z wyuczoną strategią:
            kat, V, czy_zatrzymanie = choose_action(stan, model, param_fiz)

            # zapis kroku historii:
            # phist.write(str(epizod + 1) + "  " + str(krok) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(kat) + "  " + str(V) + "\n")
            phist.write(
                "%d %d %.4f %.4f %.4f %.4f %.4f\n"
                % ((epizod + 1), krok, stan[0], stan[1], stan[2], kat, V)
            )
            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja) | (krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            stan = nowystan
        ocena_koncowa = pm.final_score(param_fiz, nowystan, czy_kolizja, krok)
        sr_ocena_koncowa += ocena_koncowa / liczba_stanow_poczatkowych
        sr_liczba_krokow = sr_liczba_krokow + krok / liczba_stanow_poczatkowych
        print(
            "w %d epizodzie ocena parkowania = %g, liczba krokow = %d"
            % (epizod, ocena_koncowa, krok)
        )

    print("srednia ocena końcowa na epizod = %g" % (sr_ocena_koncowa))
    print("srednia liczba krokow = %g" % (sr_liczba_krokow))
    phist.close()
    return sr_ocena_koncowa


park_train()