import numpy as np
import random as rnd
import csv
import igraph as ig
import time
import gzip


def random_edge_addition(graph, edge_factor):
    # adds random edges = to the number of vertices * the edge factor
    edges = []
    number_of_edges = graph.vcount()

    for a in range(number_of_edges * edge_factor):
        # generates a random vertex
        node_1 = rnd.randint(0, number_of_edges)
        node_2 = rnd.randint(0, number_of_edges)
        edges += [(node_1, node_2)]

    # adds edges to graph together at the end as is much more efficient
    return graph.add_edges(edges)


def closeness_and_degree_calculation_igraph(test_graph):
    # Calculate degree and closeness
    degree = test_graph.degree()

    print('Closeness being calculated')
    start = time.time()
    closeness = test_graph.closeness()
    end = time.time()
    print('closeness took: %f s' % (end - start))

    return closeness, degree


# Converts closeness and degree lists to average closeness for a degree with error

def closeness_and_degree_to_x_y_bins(closeness_values, degree_values):
    print('Data being binned and errors calculated')
    # convert to arrays
    closeness_values = np.array(closeness_values)
    degree_values = np.array(degree_values)

    # converts to 1/closeness
    y_val = 1 / closeness_values
    x_val = degree_values

    # Variable declarations to be used later
    y_sub_total = 0
    x_sub_total = 0
    count = 0
    y_avg = []
    x_avg = []
    y_avg_data = []
    x_avg_data = []
    y_avg_error = []
    x_avg_error = []
    error_sum_y = 0
    error_sum_x = 0
    ordered_degree = np.sort(degree_values)
    max_x_val = max(x_val)

    # Loops through all the possible degrees
    for i in range(max_x_val + 1):
        if i % 1000 == 0:
            print('Degree # %i out of %i' % (i, max_x_val))
        # for each degree adds the values to a total and appends the item to a list
        for j in range(np.size(x_val)):
            if x_val[j] == i:
                y_sub_total += y_val[j]
                y_avg_data.append(y_val[j])
                count += 1
                x_sub_total += x_val[j]
                x_avg_data.append(x_val[j])

        # if there are 5 or more items inspected then calculate averages and errors
        # if there are more than 4 degrees left to process
        if count > 4 and i < ordered_degree[-4]:

            # calculate average
            y_avg_value = y_sub_total / count
            x_avg_value = x_sub_total / count

            # calculate sum of errors
            for k in range(count):
                error_sum_y += (y_avg_data[k] - y_avg_value) ** 2
                error_sum_x += (x_avg_data[k] - x_avg_value) ** 2

            # calculate standard deviation in x and y with error propagation
            sd_y = np.sqrt((1 / (count - 1)) * error_sum_y)
            sd_x = np.sqrt(((1 / x_avg_value) ** 2) * (1 / (count - 1)) * error_sum_x)

            # append to list
            y_avg.append(y_avg_value)
            x_avg.append(x_avg_value)
            y_avg_error.append(sd_y)

            # if x error is 0 as all same degree make error very small
            if sd_x == 0:
                sd_x = 0.0000000000001
            x_avg_error.append(sd_x)

            # reset variables for next loop
            y_sub_total = 0
            x_sub_total = 0
            count = 0
            error_sum_x = 0
            error_sum_y = 0
            y_avg_data = []
            x_avg_data = []

        # if final loop then do as above
        elif i == max(x_val):

            # calculate average
            y_avg_value = y_sub_total / count
            x_avg_value = x_sub_total / count

            # calculate sum of errors
            for k in range(count):
                error_sum_y += (y_avg_data[k] - y_avg_value) ** 2
                error_sum_x += (x_avg_data[k] - x_avg_value) ** 2

            # calculate standard deviation in x and y with error propagation
            sd_y = np.sqrt((1 / (count - 1)) * error_sum_y)
            sd_x = np.sqrt(((1 / x_avg_value) ** 2) * (1 / (count - 1)) * error_sum_x)

            # append to list
            y_avg.append(y_avg_value)
            x_avg.append(x_avg_value)
            y_avg_error.append(sd_y)

            # if x error is 0 as all same degree make error very small
            if sd_x == 0:
                sd_x = 0.0000000000001
            x_avg_error.append(sd_x)

            # reset variables for next loop
            y_sub_total = 0
            x_sub_total = 0
            count = 0
            error_sum_x = 0
            error_sum_y = 0
            y_avg_data = []
            x_avg_data = []

    # convert final values and make x into log x, error propagation done above
    y = np.array(y_avg)
    x = np.log(np.array(x_avg))
    sd_y = np.array(y_avg_error)
    sd_x = np.array(x_avg_error)

    # return x, y, sd_x, sd_y
    return x, y, sd_x, sd_y


# returns the full data in the required form for the graphs
def get_full_data(close, deg):
    close = np.array(close)
    x_full = np.log(deg)
    y_full = 1 / close
    return x_full, y_full


# gz compressed file reader
def gz_graph_reader_bastard(file_name, numb_nodes):
    # creates graph and adds nodes
    new_graph = ig.Graph()
    new_graph.add_vertices(numb_nodes)

    skip_lines = 0
    edges = []
    count = 0
    nodes = {}
    next_num = 0

    # open file and decode
    with gzip.open(file_name, 'r') as f:
        for line in f:
            if count >= skip_lines:
                if count % 1000 == 0:
                    print('lines read: %i' % count)
                count += 1
                # converts to string, then separates
                line = line.decode('utf-8')
                rand_string = list(line)
                good_string = []
                middle = []
                for a in range(len(rand_string) - 1):
                    try:
                        int(rand_string[a])
                        good_string.append(rand_string[a])
                    except ValueError:
                        middle.append(a)

                num1 = good_string[0:middle[0]]
                num2 = good_string[middle[0]:len(good_string)]

                num1 = ''.join(num1)
                num2 = ''.join(num2)
                u = int(num1)
                v = int(num2)

                # checks if the node exists in the dictionary
                if num1 in nodes:
                    # if it does it finds its vertex value
                    u = nodes.get(num1)

                else:
                    # if it doesn't it gives it the next available vertex id
                    nodes[num1] = next_num
                    u = next_num
                    next_num += 1

                # repeats for the second node
                if num2 in nodes:
                    v = nodes.get(num2)
                else:
                    nodes[num2] = next_num
                    v = next_num
                    next_num += 1

                # checks if duplicate, if not adds edge to list
                if new_graph.get_eid(u, v, directed=False, error=False) == -1:
                    edges += [(u, v)]
            else:
                count += 1

    new_graph.add_edges(edges)

    # to be used if lots of un-connected graphs
    largest_graph = new_graph.clusters().giant()
    return largest_graph


# gz compressed file reader
def gz_graph_reader(file_name, numb_nodes):
    # creates graph and adds nodes
    new_graph = ig.Graph()
    new_graph.add_vertices(numb_nodes)

    skip_lines = 0
    edges = []
    count = 0

    # open file and decode
    with gzip.open(file_name, 'r') as f:
        for line in f:
            if count >= skip_lines:
                # converts to string, then separates
                line = line.decode('utf-8')

                line = line.split(' ')

                # takes the nodes of the edges
                u = int(line[0])
                v = int(line[1])
                print(u, ' ', v)

                # checks if duplicate, if not adds edge to list
                if new_graph.get_eid(u, v, directed=False, error=False) == -1:
                    edges += [(u, v)]
            else:
                count += 1

    new_graph.add_edges(edges)

    # to be used if lots of un-connected graphs
    largest_graph = new_graph.clusters().giant()
    return largest_graph


def read_database_edges(file_nom, numb_nodes):
    # creates graph and adds nodes
    new_graph = ig.Graph()
    new_graph.add_vertices(numb_nodes)
    count = 0
    number_of_lines_skip = 1
    edges = []

    # open file and take data
    with open(file_nom, newline='') as csvfile:
        line_reader = csv.reader(csvfile, delimiter=',')
        for row in line_reader:
            if count != number_of_lines_skip - 1:
                if count % 1000 == 0:
                    print('lines read: %i' % count)
                count += 1

                # takes the nodes of the edges
                u = int(row[0])
                v = int(row[1])

                # checkes if duplicate, if adds to edge list
                if new_graph.get_eid(u, v, directed=False, error=False) == -1:
                    edges += [(u, v)]
            else:
                count += 1

    new_graph.add_edges(edges)

    # to be used if lots of un-connected graphs
    largest_graph = new_graph.clusters().giant()
    return largest_graph

def read_database_edges_tsv(file_nom, numb_nodes):
    # creates graph and adds nodes
    new_graph = ig.Graph()
    new_graph.add_vertices(numb_nodes)
    count = 0
    number_of_lines_skip = 1
    edges = []
    # create dictionary to store node values
    nodes = {}
    next_num = 0

    # open file and take data
    with open(file_nom, newline='') as csvfile:
        line_reader = csv.reader(csvfile, delimiter='\t')
        for row in line_reader:
            if count >= number_of_lines_skip:
                if count % 1000 == 0:
                    print('lines read: %i' % count)
                count += 1
                if row[0] in nodes:
                    u = nodes.get(row[0])
                else:
                    nodes[row[0]] = next_num
                    u = next_num
                    next_num += 1

                if row[1] in nodes:
                    v = nodes.get(row[1])
                else:
                    nodes[row[1]] = next_num
                    v = next_num
                    next_num += 1

                # checkes if duplicate, if adds to edge list
                if new_graph.get_eid(u, v, directed=False, error=False) == -1:
                    edges += [(u, v)]
            else:
                count += 1
    new_graph.add_edges(edges)

    # to be used if lots of un-connected graphs
    largest_graph = new_graph.clusters().giant()
    return largest_graph


# gz compressed file reader
def gz_graph_reader_igraph(file_name, numbe_nodes):
    # creates graph and adds nodes
    new_graph = ig.Graph()
    new_graph.add_vertices(numbe_nodes + 1)

    # creates a dictionary to take the input values and give them the next available vertex id
    nodes = {}
    next_num = 0
    count = 0
    edges = []

    # open file and decode
    with gzip.open(file_name, 'r') as f:
        for line in f:
            # print(line)
            if count >= 0:
                # converts to string
                line = line.decode('utf-8')
                rand_string = list(line)
                # turns string into char array
                good_string = []
                middle = []

                # works out where the 2 numbers split
                for a in range(len(rand_string) - 1):
                    try:
                        int(rand_string[a])
                        good_string.append(rand_string[a])
                    except ValueError:
                        middle.append(a)

                # breaks into the 2 numbers
                num1 = good_string[0:middle[0]]
                # num2 = good_string[middle[0]:len(rand_string)-1]
                num2 = good_string[middle[0]:middle[1] - 1]

                # joins the char arrays to string
                num1 = ''.join(num1)
                num2 = ''.join(num2)
                # print(num1, ', ', num2)

                # checks if the node exists in the dictionary
                if num1 in nodes:
                    # if it does it finds its vertex value
                    u = nodes.get(num1)

                else:
                    # if it doesn't it gives it the next available vertex id
                    nodes[num1] = next_num
                    u = next_num
                    next_num += 1

                # repeats for the second node
                if num2 in nodes:
                    v = nodes.get(num2)
                else:
                    nodes[num2] = next_num
                    v = next_num
                    next_num += 1
                if u > num_nodes or v > num_nodes:
                    print(num1, num2)
                    print(u, v)
                    print(next_num)

                # checks if duplicate, if not makes edge
                if new_graph.get_eid(u, v, directed=False, error=False) == -1:
                    edges += [(u, v)]

                # keep track of progress
                if count % 100000 == 0:
                    print(count)
                count += 1
            else:
                count += 1

    new_graph.add_edges(edges)
    # to be used if lots of un-connected graphs
    largest_graph = new_graph.clusters().giant()
    return largest_graph


def save_data(file_name, deg_full, clo_full, deg_avg, clo_avg, deg_sd, clo_sd, edge_swap):
    if edge_swap:
        file_name = file_name + 'full_set_swapped.csv'
    else:
        file_name = file_name + 'full_set.csv'
    names = ['Full Degree', 'Full Closeness', 'Avg Degree', 'Avg Closeness', 'Degree SD', 'Closeness SD']
    datas = [deg_full, clo_full, deg_avg, clo_avg, deg_sd, clo_sd]
    with open(file_name, 'w') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        for a in range(6):
            datas[a] = list(datas[a])
            data_writer.writerow((names[a], datas[a]))


def randomise_graph(graph, swap_edges, edge_addition):
    new_graph = graph
    if edge_addition > 0:
        print('Adding edges')
        new_graph = random_edge_addition(new_graph, edge_addition)

    if swap_edges:
        print('Swapping edges')
        num_edges = new_graph.ecount()
        ig.summary(new_graph)
        new_graph.rewire(num_edges * 10, mode='loops')
        new_graph = new_graph.clusters().giant()
        ig.summary(new_graph)
        print(len(new_graph.decompose()))

    final_graph = new_graph
    return final_graph


def get_data(file_name, number_nodes, edge_addition_factor, ed_swap):
    # Create graph from file with edges
    # graph = gz_graph_reader_bastard(file_name, number_nodes)
    # graph = gz_graph_reader(file_name, number_nodes)
    # graph = read_database_edges_tsv(file_name, number_nodes)
    graph = read_database_edges(file_name, number_nodes)
    # graph = gz_graph_reader_igraph(file_name, number_nodes)

    # Add edges = number of nodes * factor
    # random_edge_addition(graph, edge_addition_factor)
    graph = randomise_graph(graph, ed_swap, edge_addition_factor)

    # Calculates closeness and degree for every node in the graph
    closeness, degree = closeness_and_degree_calculation_igraph(graph)

    # Converts the closeness and degree values to binned data with errors
    degree_avg, closeness_avg, degree_sd, closeness_sd = closeness_and_degree_to_x_y_bins(closeness, degree)

    # Keeps a full record of closeness and degree in the right format
    degree_full, closeness_full = get_full_data(closeness, degree)

    # Saves the data to a csv file
    save_data(file_name, degree_full, closeness_full, degree_avg, closeness_avg, degree_sd, closeness_sd, ed_swap)
    print('Data saved successfully')


file = 'RealNetworks/deezer_europe/deezer_europe_edges.csv'
# file = 'RealNetworks/gplus/gplus_combined.txt.gz'
num_nodes = 28281
edge_addition_multiplier = 0
edge_swapping = True

get_data(file, num_nodes, edge_addition_multiplier, edge_swapping)
