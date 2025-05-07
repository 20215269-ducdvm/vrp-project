#!/usr/bin/python
"""draw

Usage:
    draw.py <_problem_instance>
"""
import matplotlib.pyplot as plt
import sys
import os
import mplcursors
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem_parser import parse_problem


def draw(problem_instance):
    coordinates = problem_instance['coords']
    n = problem_instance['_location_number']  # number of locations (depot + customers)
    v = problem_instance['_vehicle_number']  # number of vehicles
    name = problem_instance['type']  # type of the problem instance
    q = problem_instance['demand']  # demand of each customer
    s = problem_instance['service_duration']  # service time at each customer
    t = problem_instance['t']  # travel time between customers
    e = problem_instance['time_window_start']  # start of time window for each customer
    l = problem_instance['time_window_end']  # end of time window for each customer

    x_values = [point[0] for point in coordinates]
    y_values = [point[1] for point in coordinates]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', s=10)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Coordinate Points')

    plt.figtext(0.15, 0.95, f"Name: {name}", ha='left', va='center')
    plt.figtext(0.15, 0.92, f"Number of Vehicles: {v}", ha='left', va='center')
    plt.figtext(0.15, 0.89, f"Capacity: {problem_instance['capacity']}", ha='left', va='center')

    plt.annotate("Depot", (x_values[0], y_values[0]), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    scatter = plt.scatter(x_values, y_values, color='blue', s=10)

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        if index == 0:
            sel.annotation.set(text=f"Depot\nLocation: {coordinates[index]}\n"
                                    f"Due date: {l[index]}\n")

        else:
            sel.annotation.set(text=f"Customer: {index}\n"
                                    f"Location: {coordinates[index]}\n"
                                    f"Demand: {q[index]}\n"
                                    f"Time Window: [{e[index]}, {l[index]}]\n"
                                    f"Service Time: {s[index]}")

        sel.annotation.set_position((10, 0))

    def on_click(event):
        if event.inaxes is not None:
            for sel in cursor.selections:
                if sel.annotation.get_visible():
                    bbox = sel.annotation.get_bbox_patch().get_extents()
                    if not (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
                        sel.annotation.set_visible(False)
                        plt.draw()

    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()
