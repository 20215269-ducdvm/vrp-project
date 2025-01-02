#!/usr/bin/python
"""hsa

Usage:
    hsa.py <problem_instance> --hms=<hms> --hmcr=<hmcr> --parmax=<par> --parmin=<parmin> --ni=<ni>

Options:
    --hms=<hms>     Harmony memory size e.g. 10, 20, 30...
    --hmcr=<hmcr>   Harmony memory consideration rate e.g. 0.6, 0.7, 0.8
    --ni=<ni>       Number of improvisations e.g. 500, 1000, 2000
    --parxmax=<parmax>  Maximal pitch adjustment rate e.g. 0.9
    --parxmin=<parmin>  Minimal pitch adjustment rate e.g. 0.3

"""


def parse_line(line):
    return [int(item) for item in line.split()]


def get_problem_name(lines):
    return lines[0].rstrip()


def get_vehicle_number_and_capacity(lines):
    # in Solomon problems all the vehicles have the same capacity
    return parse_line(lines[4])


def not_empty_line(line):
    return line.split() != []


def get_customers(lines):
    return [parse_line(customer_line) for customer_line in lines[9:] if not_empty_line(customer_line)]


from math import sqrt


def distance(coords1, coords2):
    (x1, y1) = coords1
    (x2, y2) = coords2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_coords(customer_list, i):
    # 0 is depot
    # i is number of client beginnging from one
    return customer_list[i]['coords']


def parse_problem_lines(lines):
    """return dictionary containing problem variables"""
    problem_instance = {}
    [vehicle_number, capacity] = get_vehicle_number_and_capacity(lines)
    problem_instance['name'] = get_problem_name(lines)
    problem_instance['vehicle_number'] = vehicle_number
    problem_instance['capacity'] = capacity
    # naming as in "Harmony Search Algorith for Vehicle Rougint Problem with Time Windows"
    # Esam Taha Yassen et al.
    # http://scialert.net/qredirect.php?doi=jas.2013.633.638&linkid=pdf
    # t is travel time matrix t[0..customer_number][0..customer_number]
    # denoting travel time between t[i][j] (it is symmetric)
    # n is customer number (fixed to 100 in Solomon problem set)
    n = 1
    # q[i] is demand of customer c[i]
    # Q[k] is capacity of k th vevicle
    # s[i] is service duration of consumer c[i]
    # e[i] is starting time window for customer c[i]
    # l[i] is end time window for customer c[i]
    customer_list = []
    for customer in get_customers(lines):
        [number, xcoord, ycoord, demand, ready, due, servicetime] = customer
        customer_dict = {'number': number, 'coords': (xcoord, ycoord),
                         'demand': demand, 'ready': ready, 'due': due, 'servicetime': servicetime}
        customer_list.append(customer_dict)
    n = len(customer_list)
    # initialize variables
    q = [0] * n  # Initialize list of zeros with length n
    s = [0] * n
    e = [0] * n
    l = [0] * n
    t = []
    # number of clients plus depot
    t = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        q[i] = customer_list[i]['demand']
        s[i] = customer_list[i]['servicetime']
        e[i] = customer_list[i]['ready']
        l[i] = customer_list[i]['due']
    for i in range(n):
        for j in range(n):
            value = distance(get_coords(customer_list, i), get_coords(customer_list, j))
            truncated_value = int(value * 10) / 10
            t[i][j] = t[j][i] = truncated_value

    problem_instance['demand'] = q
    problem_instance['service_duration'] = s
    problem_instance['time_window_start'] = e
    problem_instance['time_window_end'] = l
    problem_instance['t'] = t
    problem_instance['customer_number'] = n

    return problem_instance


def parse_problem(path_to_file):
    """read file and return dictionary containing problem variables"""
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        return parse_problem_lines(lines)


# create random solution
from random import shuffle


def pfih(problem_instance):
    n = problem_instance['customer_number']
    v = problem_instance['vehicle_number']
    x = []
    for i in range(n):
        tmp1 = []
        for j in range(n):
            tmp2 = []
            for k in range(v):
                tmp2.append(0)
            tmp1.append(tmp2)
        x.append(tmp1)

    y = []
    for i in range(n):
        tmp1 = []
        for k in range(v):
            tmp1.append(0)
        y.append(tmp1)

    customers_to_serve = list(range(n))
    customers_to_serve.pop(0)
    shuffle(customers_to_serve)

    # take first customer and add to first route
    k = 0
    c = customers_to_serve.pop(0)
    x[0][c][k] = 1
    while customers_to_serve:
        # put c in the first route
        pass


def check_constraints(x, problem_instance):
    n = problem_instance['customer_number']  # number of customers
    v = problem_instance['vehicle_number']  # number of vehicles
    y = [[0 for _ in range(v)] for _ in range(n)]  # y[i][k] = 1 if customer i is served by vehicle k
    q = problem_instance['demand']  # demand for each customer
    a = [0] * n  # arrival time for each customer
    W = [0] * n  # waiting time for each customer
    s = problem_instance['service_duration']  # service duration for each customer
    t = problem_instance['t']  # travel time matrix
    e = problem_instance['time_window_start']  # time window start for each customer
    l = problem_instance['time_window_end']  # time window end for each customer

    # CONSTRAINTS FOR CVRP

    # additional constraint: i != j, meaning that vehicle must travel to different customers
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                for k in range(v):
                    if x[i][j][k] == 1:
                        return False

    ### Only one vehicle arriving at each customer and only one vehicle departing from each customer
    # This implies that each customer is served by exactly one vehicle
    # (2) Track the number of vehicles arriving at each customer
    for k in range(v):
        for j in range(n):
            for i in range(n):
                y[j][k] = y[j][k] + x[i][j][k]

    # (3) Track the number of vehicles departing from each customer
    for k in range(v):
        for i in range(n):
            total = 0
            for j in range(n):
                total = total + x[i][j][k]
            if total != y[i][k]:  # departure from customer i is not equal to arrival at customer i
                # print("3")
                return False

    # (4) vehicle capacity is not exceeded
    for k in range(v):
        total = 0
        for i in range(n):
            total = total + y[i][k] * q[i]
        if total > problem_instance['capacity']:
            # print("4")
            return False
    # (5) each customer is served only once
    for i in range(1, n):
        total = 0
        for k in range(v):
            total = total + y[i][k]
        if total != 1:
            # print("5")
            return False
    # (6) every vehicle starts from depot
    for k in range(v):
        if y[0][k] != 1:
            # print("6")
            return False

    # CONSTRAINTS FOR TIME WINDOWS

    location_count = 0
    for k in range(v):
        current_location = 0
        while True:
            next_location = None

            # find next location
            for j in range(n):
                if x[current_location][j][k] == 1:
                    next_location = j
                    break
            location_count += 1  # location_count at depot is duplicated v times since every vehicle starts from depot

            if next_location is None or next_location == 0:
                break

            # (7) calculate arrival time at next location
            a[next_location] = a[current_location] + W[current_location] + s[current_location] + t[current_location][
                next_location]

            # (9) waiting time is either service start time - arriving time or 0
            # vehicle can arrive at a customer before time window, but it has to wait until the time window starts
            W[next_location] = max(e[next_location] - a[next_location], 0)

            # (8) customer cannot be served after end of time window
            if a[next_location] > l[next_location]:
                # print("Time window constraint violated at customer", next_location)
                return False
            current_location = next_location

    if location_count - v + 1 != n:
        # print("Not all customers are served")
        return False

    return True


# not finished
def f(t, x, n, v):
    """quality function, that we want to minimize - total travelled distance"""
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(v):
                total = t[i][j] * x[i][j][k]


from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__)
    problem_instance = parse_problem(arguments['<problem_instance>'])
    hms = int(arguments['--hms'])
    hmcr = float(arguments['--hmcr'])
    parmax = float(arguments['--parmax'])
    parmin = float(arguments['--parmin'])
    ni = int(arguments['--ni'])
    generation = 0
    print(parmax, parmin, ni)
    # steps defined in "Harmony Search Algorith for Vehicle Rougint Problem with Time Windows"
    # Esam Taha Yassen et al.
    # http://scialert.net/qredirect.php?doi=jas.2013.633.638&linkid=pdf
    print(pfih(problem_instance))
