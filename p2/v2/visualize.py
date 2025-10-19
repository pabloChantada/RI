
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        avg_fitness = np.array(statistics.get_fitness_mean())
        stdev_fitness = np.array(statistics.get_fitness_stdev())

        plt.plot(generation, avg_fitness, 'b-', label="average")
        plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
        plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
        plt.plot(generation, best_fitness, 'r-', label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale('symlog')

        plt.savefig(filename)
        if view:
            plt.show()

        plt.close()

    @staticmethod
    def plot_species(statistics, view=False, filename='speciation.svg'):
        """ Visualizes speciation throughout evolution. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        species_sizes = statistics.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.savefig(filename)

        if view:
            plt.show()

        plt.close()

    @staticmethod
    def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
                 node_colors=None, fmt='svg'):
        """ Receives a genome and draws a neural network with arbitrary topology. """
        if graphviz is None:
            warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
            return

        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'}

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs_output = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
            dot.node(name, _attributes=node_attrs_output)

        # Si prune_unused es True, solo mostrar nodos que están conectados
        if prune_unused:
            # Encontrar nodos que están en uso (tienen conexiones activas)
            used_nodes = set()
            for cg in genome.connections.values():
                if cg.enabled:
                    used_nodes.add(cg.key[0])
                    used_nodes.add(cg.key[1])
        else:
            used_nodes = set(genome.nodes.keys())

        # Dibujar nodos ocultos
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {'style': 'filled',
                     'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)

        # Dibujar conexiones
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input_node, output_node = cg.key
                
                # Si estamos podando y el nodo no está en uso, saltar
                if prune_unused and not cg.enabled:
                    continue
                    
                a = node_names.get(input_node, str(input_node))
                b = node_names.get(output_node, str(output_node))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

        dot.render(filename, view=view)

        return dot

