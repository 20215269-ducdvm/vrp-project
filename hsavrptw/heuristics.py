from math import atan2, pi

from problem_parser import distance


def pfih(problem_instance):
    """
        Push Forward Insertion Heuristic
        PFIH starts with an empty set of routes and iteratively inserts customers into existing routes or creates new routes as long as constraints are met.
        This strategy is used to initialize a solution in the harmony memory.
    """
    coords = problem_instance['coords']  # coordinates of each customer
    capacity = problem_instance['capacity']  # total capacity
    n = problem_instance['location_number']  # number of locations (depot + customers)
    v = problem_instance['vehicle_number']  # number of vehicles
    y = [[0 for _ in range(v)] for _ in
              range(n)]  # 1 if customer is visited by vehicle k, 0 otherwise
    q = problem_instance['demand']  # demand of each customer
    a = [0] * n  # arrival time at each customer
    W = [0] * n  # Waiting time at each customer
    s = problem_instance['service_duration']  # service time at each customer
    t = problem_instance['t']  # travel time between customers
    e = problem_instance['time_window_start']  # start of time window for each customer
    l = problem_instance['time_window_end']  # end of time window for each customer

    # Step 1: Begin with an Empty Route starting from the depot
    # Start a new route from the central depot C_0. Set the counter r = 1
    # No customers are assigned, vehicle's capacity and time constraints are at starting values

    solution = [] # list of routes. expected outcome [(0, [0, 1, 2, 3, 0]), (1, [0, 4, 5, 0]), (2, [0, 6, 0])]
    remaining_capacity = capacity # remaining capacity of the vehicles
    route_counter = 0 # route counter (vehicle counter)
    visited = [] * n # boolean list to check if customer has been visited


    # Step 2: Check if all customers have been routed
    # - if yes, proceed to step 8
    # - otherwise, compute the cost for each unrouted customer, sort them in ascending order and select the customer with the lowest cost

    cost = [] # list to store (customer_idx, cost) tuples
    def all_visited(arr):
        for idx in arr:
            if not idx:
                return False
        return True

    def polar_coord(idx):
        p = atan2(coords[idx][1] - coords[0][1], coords[idx][0] - coords[0][0]) * 180 / pi
        if p < 0:
            p += 360
        return p

    def compute_initial_cost(customer_idx):
        alpha = 0.7
        beta = 0.1
        gamma = 0.2
        return -alpha * t[0][customer_idx] + beta * l[customer_idx] + gamma * (polar_coord(customer_idx) / 360 * t[0][customer_idx])

    while True:
        if all_visited(visited):
            break
        route = [0]
        for i in range(1, n):
            if visited[i]:
                continue
            cost[i].append((i, compute_initial_cost(i)))
        # sort the initial costs
        cost = sorted(cost, key=lambda x: x[1])
        # Step 3: Select the first customer with the lowest cost and meet the constraints
        # - if the vehicle's capacity is not exceeded, add the customer to the route
        for c in cost:
            if q[c[0]] <= remaining_capacity and a[route[-1]] + t[route[-1]][c[0]] + s[c[0]] <= l[c[0]]:
                route.append(c[0])
                visited[c[0]] = True
                # Step 4: Add the selected customer to the current route, and update that route's capacity and schedule
                remaining_capacity -= q[c[0]] # update the remaining capacity
                a[c[0]] = a[route[-1]] + t[route[-1]][c[0]] + s[c[0]] # update arrival time at customer
                W[c[0]] = max(0, e[c[0]] - a[c[0]]) # update waiting time at customer
                break
        # Step 5: Compute the insertion costs for unrouted customers
        # For all unrouted customers j: For all edges {k, l} in the current route,
        # compute the cost of inserting each of the unrouted customers
        # between k and l. Assume the insertion cost simplifies to additional distance
        # between k and j plus distance between j and l minus the distance between k and l.
        min_cost = float('inf')
        best_insertion = None
        unrouted_customers = [i for i in range(1, n) if not visited[i]]
        for j in unrouted_customers:
            if q[j] > remaining_capacity:
                continue
            for k in range(len(route) - 1):
                i = route[k]
                l = route[k + 1]
                insertion_cost = t[i][j] + t[j][l] - t[i][l]
                if insertion_cost < min_cost and a[i] + t[i][j] + s[j] <= l[j]:
                    min_cost = insertion_cost
                    best_insertion = (i, l, j)

        # Step 6: Select and Insert the Best Feasible Customer
        # - Choose the unrouted customer j and edge {k*, l*} with the least insertion cost, ensuring time and capacity constraints.
        #   - If $j$  exists, update the route, return to Step 5
        #   - If none exists, proceed to Step 7
    return solution



    # Step 8:

def lambda_interchange(vector):
    """
    After having generated a solution by pfih, the lambda interchange heuristic will be used to generate the rest solutions in the HM.
    """
    pass
