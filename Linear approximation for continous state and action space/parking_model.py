import numpy as np
import pdb

# The model of car movement in the parking area (street + parking place)
# Input:
# state - xs,ys - coordinates of a center of a car, alpha - car orientation angle
# wheel_turn_angle - front wheel turn angle
# V - car velocity (positive - forward, nagative - backward movement)
# Output:
# state_next - next state after time dt
# rotation_center - coordinates of a point around which the car rotates
# PLEASE, DO NOT CHANGE ANYTHING IN THIS FILE!


class GlobalVar:
    # sizes in meters:
    place_width = 8.6  # width of a parking place
    park_depth = 3.0  # depth of a parking place
    street_width = 7
    street_length = 26  # length of a fragment of a street
    car_width = 2.3
    car_length = 5.2
    front_axis_dist = 1.6  # distance from front axle to front bumper of a car
    back_axis_dist = 0.45  # distance from back axle to back bumper of a car
    dt = 0.1  # time between simulation steps in seconds
    Vmod = 2.0  # absolute value of velocity [m/s]
    wheel_turn_angle_max = np.pi / 4  # the maximum turning angle of the wheels [rad]
    if_side_parking_place = True
    max_number_of_steps = 500


def park_save(filename, param):
    pli = open("param.txt", "w")
    pli.write("szer_park = " + str(param.place_width) + "\n")
    pli.write("gleb_park = " + str(param.park_depth) + "\n")
    pli.write("szer_uli = " + str(param.street_width) + "\n")
    pli.write("dl_uli = " + str(param.street_length) + "\n")
    pli.write("szer_auta = " + str(param.car_width) + "\n")
    pli.write("dl_auta = " + str(param.car_length) + "\n")
    pli.write("odl_osi_prz = " + str(param.front_axis_dist) + "\n")
    pli.write("odl_osi_tyl = " + str(param.back_axis_dist) + "\n")
    pli.write("dt = " + str(param.dt) + "\n")
    pli.write("Vmod = " + str(param.Vmod) + "\n")
    pli.write("max_kat = " + str(param.wheel_turn_angle_max) + "\n")
    pli.write("park_boczne = %d\n" % (param.if_side_parking_place))
    pli.write("liczba_krokow_maks = %d\n" % (param.max_number_of_steps))
    pli.close()


# Localication of corners of a car based on current state
# PLEASE, DO NOT CHANGE ANYTHING IN THIS FILE!
def corners_of_car(state, var_global):
    car_width = var_global.car_width
    car_length = var_global.car_length

    x, y, alfa = state

    xA = x - car_length / 2 * np.cos(alfa) - car_width / 2 * np.sin(alfa)
    yA = y - car_length / 2 * np.sin(alfa) + car_width / 2 * np.cos(alfa)

    xB = xA + car_width * np.sin(alfa)
    yB = yA - car_width * np.cos(alfa)

    xD = xA + car_length * np.cos(alfa)
    yD = yA + car_length * np.sin(alfa)

    xC = xB + car_length * np.cos(alfa)
    yC = yB + car_length * np.sin(alfa)

    X = np.array([xA, xB, xC, xD])
    Y = np.array([yA, yB, yC, yD])
    return X, Y


# Model parkującego samochodu - na wejściu stan (położenie auta), kąt skrętu kół i prędkość (dodatnia - do przodu,
# ujemna - do tyłu):
# zwraca nowy stan + środek obrotu + informację o kolizji
def model_of_car(var_global, state, wheel_turn_angle, V):
    dt = var_global.dt
    place_width = var_global.place_width
    park_depth = var_global.park_depth
    street_width = var_global.street_width
    street_length = var_global.street_length
    car_width = var_global.car_width
    car_length = var_global.car_length
    front_axis_dist = var_global.front_axis_dist
    back_axis_dist = var_global.back_axis_dist
    wheel_turn_angle_max = var_global.wheel_turn_angle_max

    number_of_iterations = 5  # number of iterations if car hit an obstacle (as more as better approximation of collision moment)
    if np.abs(wheel_turn_angle) > wheel_turn_angle_max:
        wheel_turn_angle = np.sign(wheel_turn_angle) * wheel_turn_angle_max
    if np.abs(V) > var_global.Vmod:
        V = V * var_global.Vmod / np.abs(V)

    initial_state = state
    if_end = False
    iter = 1
    collision_detected = False
    while if_end == False:
        xs, ys, alpha = initial_state  # center of car coordinates
        # alpha - car orientation angle, alpha = 0, when car longitudinal axis || to x axis and car is directed to the right

        # the difference between car geometrical center and center between
        # axes:
        ds = (front_axis_dist - back_axis_dist) / 2

        # x,y - coordinates of rectangular area between car axles in accordance
        # to ultimate position (x=0,y=0 in the center of parking place):
        x = xs - ds * np.cos(alpha)
        y = ys - ds * np.sin(alpha)

        # 1. Calculation of new position of a center of rectangular area
        # between car axles
        d_osi = car_length - front_axis_dist - back_axis_dist  # distance between axles
        rotation_center = [0, 0]
        if abs(wheel_turn_angle) < 1e-6:
            xn = x + V * dt * np.cos(alpha)
            yn = y + V * dt * np.sin(alpha)
            alpha_next = alpha
        else:
            a = d_osi / np.tan(wheel_turn_angle)
            Ro = np.sqrt(
                d_osi * d_osi / 4 + ((abs(a) + car_width / 2) ** 2)
            )  # radius of car rotation
            tau = np.sign(wheel_turn_angle) * alpha + np.arcsin(d_osi / 2 / Ro)
            rotation_center = [
                x - Ro * np.sin(tau),
                y + np.sign(wheel_turn_angle) * Ro * np.cos(tau),
            ]  # center of rotation
            gama = V * dt / Ro  # wheel_turn_angle
            xn = x + Ro * (np.sin(gama + tau) - np.sin(tau))
            yn = y + np.sign(wheel_turn_angle) * Ro * (np.cos(tau) - np.cos(gama + tau))
            alpha_next = alpha + np.sign(wheel_turn_angle) * gama
            if abs(alpha_next) > np.pi:
                alpha_next = alpha_next - np.sign(alpha_next) * np.pi * 2
        xsn = xn + ds * np.cos(
            alpha_next
        )  # new position of geometrical center of a car
        ysn = yn + ds * np.sin(alpha_next)
        state_next = [xsn, ysn, alpha_next]

        # 2. Wyznaczenie polozenia naroznikow model_of_caru oraz naroznikow parkingu
        X, Y = corners_of_car(state_next, var_global)

        # Narozniki parkingu (wspolrzedne punktow w kolejnych kolumnach):
        # tylko wystajace:
        Xp = np.array([-place_width / 2, place_width / 2])
        Yp = np.array([car_width / 2, car_width / 2])

        # 3. Sprawdzenie, czy model_of_car nie zahacza o przeszkode. Jesli tak, to
        #    cofniecie do momentu zahaczenia:

        # W - Czy narozniki auta nie przekroczyly granic parkingu
        W = (
            (X >= -street_length / 2)
            & (X <= street_length / 2)
            & (Y <= street_width + car_width / 2)
            & (
                (Y >= car_width / 2)
                | (
                    (Y >= car_width / 2 - park_depth)
                    & (X >= -place_width / 2)
                    & (X <= place_width / 2)
                )
            )
        )

        # Wp - Czy naroznik parkingu nie znajduje sie 'wewnatrz' auta
        Wp = np.zeros(shape=[4])
        for i in range(len(Xp)):
            v_kr_X = np.concatenate([X[1:], [X[0]]]) - X
            v_kr_Y = (
                np.concatenate([Y[1:], [Y[0]]]) - Y
            )  # wsp. wektorow krawedzi auta ( w kolumnach)
            v_kr = np.array([v_kr_X, v_kr_Y]).transpose()

            v_pkt_X = Xp[i] - X
            v_pkt_Y = Yp[i] - Y  # wsp. wektorow od wierzcholkow auta do punktu (xp,yp)
            v_pkt = np.array([v_pkt_X, v_pkt_Y]).transpose()  # wektory w kolumnach
            ilo_weks = np.cross(v_kr, v_pkt)  # iloczyny wektorowe v_kr(j) x v_pkt(j)

            # ilo_weks = cross([v_kr; zeros(1, 4)], [v_pkt; zeros(1, 4)])  # iloczyny wektorowe v_kr(j) x v_pkt(j)
            # Wp(i) = (sum(ilo_weks(end,:) > 0) < 4)  # jesli wszystkie dodatnie, to znaczy ze punkt lezy wewnatrz auta
            # suma(i) = sum(ilo_weks(end,:))
            Wp[i] = np.sum(ilo_weks > 0) == 4

        # .pdb.set_trace()

        if (np.sum(W) < 4) | (np.sum(Wp) > 0):  # obstacle detected
            state_next = initial_state
            collision_detected = True
            V = V / 2
        elif iter > 1:  # obstacle not detected, but was detected after longer way
            initial_state = state_next
            V = V / 2
        else:  # obstacle not detected
            if_end = True

        if iter >= number_of_iterations:
            if_end = True
        iter = iter + 1

    return state_next, rotation_center, collision_detected


# returns 10 if car is parked perfectly without collision and using not so much steps
def final_score(phis_params, state, if_collision, num_of_steps):

    x = state[0]
    y = state[1]
    angle = state[2]

    distance = np.sqrt(x * x + y * y)

    angle_reduced = 0
    if phis_params.if_side_parking_place:
        if np.abs(angle) > np.pi / 2:
            angle_reduced = np.pi - np.abs(angle)
        else:
            angle_reduced = np.abs(angle)
    else:
        angle_reduced = np.abs(np.abs(angle) - np.pi / 2)

    rational_num_of_steps = (
        phis_params.park_depth + phis_params.street_width + phis_params.street_length
    ) / (phis_params.Vmod * phis_params.dt)
    excess_step_num = max(num_of_steps - rational_num_of_steps, 0)

    score = (
        10
        / (1 + distance)
        / (1 + angle_reduced * 2)
        / (1 + int(if_collision))
        / (1 + excess_step_num / rational_num_of_steps)
    )

    return score


def random_initial_states(phis_params, num_of_states):
    car_min_size = min(phis_params.car_width, phis_params.car_length)
    created_states = []
    while len(created_states) < num_of_states:
        x = (
            2
            * (np.random.rand() - 0.5)
            * (phis_params.street_length / 2 - car_min_size)
        )
        y = (
            phis_params.street_width / 2
            + phis_params.car_width / 2
            + 2
            * (np.random.rand() - 0.5)
            * (phis_params.street_width / 2 - car_min_size)
        )
        angle = 2 * (np.random.rand() - 0.5) * np.pi
        state = [x, y, angle]
        _, _, if_collision = model_of_car(phis_params, state, 0, 0)
        if if_collision == False:
            created_states.append(state)
    return np.array(created_states)