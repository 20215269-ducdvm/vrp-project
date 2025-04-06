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
import math


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


def distance(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def truncate_to_decimal(value, decimal_places=1):
    """Truncate a value to a specific number of decimal places without rounding. Default is 1 decimal place."""
    factor = 10 ** decimal_places
    return value / factor

def multiply_and_floor(value, decimal_places=1):
    factor = 10 ** decimal_places
    return math.floor(value * factor)

def get_coords(customer_list, i):
    # 0 is depot
    # i is number of client beginning from one
    return customer_list[i]['coords']

def parse_problem_lines(lines):
    """return dictionary containing problem variables"""
    problem_instance = {}
    [vehicle_number, capacity] = get_vehicle_number_and_capacity(lines)
    problem_instance['name'] = get_problem_name(lines)
    problem_instance['vehicle_number'] = vehicle_number
    problem_instance['capacity'] = capacity
    # naming as in "Harmony Search Algorith for Vehicle Routing Problem with Time Windows"
    # Esam Taha Yassen et al.
    # http://scialert.net/qredirect.php?doi=jas.2013.633.638&linkid=pdf
    # t is travel time matrix t[0..location_number][0..location_number]
    # denoting travel time between t[i][j] (it is symmetric)
    # n is location number (depot + customers) (fixed to 101 in Solomon problem set)
    # q[i] is demand of customer c[i]
    # Q[k] is capacity of k th vehicle
    # s[i] is service duration of consumer c[i]
    # e[i] is starting time window for customer c[i]
    # l[i] is end time window for customer c[i]
    # coords[i] is the coordinates of customer c[i]
    customer_list = []
    coords = []
    for customer in get_customers(lines):
        [number, xcoord, ycoord, demand, ready, due, service_time] = customer
        coords.append((xcoord, ycoord))
        customer_dict = {'number': number, 'coords': (xcoord, ycoord),
                         'demand': demand, 'ready': ready, 'due': due, 'service_time': service_time}
        customer_list.append(customer_dict)

    n = len(customer_list)

    problem_instance['location_number'] = n
    problem_instance['coords'] = coords

    # initialize variables
    q = [0] * n  # Initialize list of zeros with length n
    s = [0] * n
    e = [0] * n
    l = [0] * n
    
    # number of clients plus depot
    t = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        q[i] = customer_list[i]['demand']
        s[i] = customer_list[i]['service_time']
        e[i] = customer_list[i]['ready']
        l[i] = customer_list[i]['due']        
    for i in range(n):
        for j in range(n):            
            value = distance(get_coords(customer_list, i), get_coords(customer_list, j))
            t[i][j] = t[j][i] = multiply_and_floor(value) # to avoid floating point arithmetic errors like 1.0000000000000002

    problem_instance['demand'] = q
    problem_instance['service_duration'] = s
    problem_instance['time_window_start'] = e
    problem_instance['time_window_end'] = l
    problem_instance['t'] = t

    return problem_instance


def parse_problem(path_to_file):
    """read file and return dictionary containing problem variables"""
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        return parse_problem_lines(lines)

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
