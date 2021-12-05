import random, yaml, math, csv, os
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

""" HoI4 NSB damage dealing simulator

Copyright 2021 Filip Stefanak, filip.stefanak@gmail.com
"""

class Params:
    def __init__(self, data, **kwargs) -> None:
        def value(name):
            return kwargs.get(name) or data[name]
        self.attacker_width = value('attacker_width')
        self.defender_width = value('defender_width')
        self.combat_width = value('combat_width')
        self.attack_per_width = value('attack_per_width')
        self.attack_base = value('attack_base')
        self.defense_per_width = value('defense_per_width')
        self.defense_base = value('defense_base')
        self.effective_coordination = value('effective_coordination')
        self.overwidth_penalty_rate = value('overwidth_penalty_rate')
        self.overwidth_penalty_max = value('overwidth_penalty_max')
        self.stacking_limit_per_width = value('stacking_limit_per_width')
        self.stacking_penalty_per_division = value('stacking_penalty_per_division')

        self.attackers_count = Params.units_per_front(
            self.combat_width, self.attacker_width, self.overwidth_penalty_rate, self.overwidth_penalty_max)
        self.defenders_count = Params.units_per_front(
            self.combat_width, self.defender_width, self.overwidth_penalty_rate, self.overwidth_penalty_max)

        self.attack_overwidth_factor = Params.overwidth_penalty(
            self.combat_width, self.attacker_width, self.attackers_count, self.overwidth_penalty_rate)
        self.defense_overwidth_factor = Params.overwidth_penalty(
            self.combat_width, self.defender_width, self.defenders_count, self.overwidth_penalty_rate)

        self.attack_overstack_factor = Params.overstack_penalty(
            self.combat_width, self.attackers_count, self.stacking_limit_per_width, self.stacking_penalty_per_division)
        self.defense_overstack_factor = Params.overstack_penalty(
            self.combat_width, self.defenders_count, self.stacking_limit_per_width, self.stacking_penalty_per_division)

        self.attack = (
                (self.attack_base + (self.attacker_width * self.attack_per_width))
                * self.attack_overwidth_factor * self.attack_overstack_factor)
        self.defense = (
                (self.defense_base + (self.defender_width * self.defense_per_width))
                * self.defense_overwidth_factor * self.defense_overstack_factor)

        self.target_pool_size = min(max(1, (2 * self.attacker_width) // self.defender_width), self.defenders_count)

        self.focused_attacks = self.effective_coordination * self.attack
        self.spread_attacks = math.floor(((1 - self.effective_coordination) * self.attack) / self.target_pool_size)

    @staticmethod
    def units_per_front(combat_width, unit_width, penalty_rate, penalty_max):
        max_overflow = (1 + (penalty_max / penalty_rate)) * combat_width
        return min(math.ceil(combat_width / unit_width), math.floor(max_overflow / unit_width))

    @staticmethod
    def overwidth_penalty(combat_width, unit_width, unit_count, penalty_rate):
        overflow = max(((unit_width * unit_count / combat_width) - 1), 0)
        return 1 - (overflow * penalty_rate)

    @staticmethod
    def overstack_penalty(combat_width, unit_count, limit_per_width, penalty_per_division):
        overflow = max(unit_count - math.floor(combat_width * limit_per_width), 0)
        return 1 - (overflow * penalty_per_division)

    @staticmethod
    def load_base(**kwargs):
        with open(r'./base_params.yaml') as file:
            return Params(yaml.load(file, Loader=yaml.FullLoader), **kwargs)


class Simulation:
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def simulate(self, params, rounds, value_function=None):
        return self.process_rounds(self.simulate_round, params, rounds, value_function)

    def damage_received(self, attacks_received, params):
        return [(attacks + (max(0, attacks - params.defense) * 3)) / 10 for attacks in attacks_received]

    def process_rounds(self, simulate_round_function, params, rounds, value_function=None):
        self.counter += 1
        mean_result = sum((sum(simulate_round_function(params)) for _ in range(rounds))) / rounds
        if value_function is not None:
            return value_function(mean_result, params)
        return mean_result

    def simulate_round(self, params: Params):
        attacks_received = [0] * params.defenders_count
        spread_attacks = params.spread_attacks
        focused_attacks = params.focused_attacks

        for attacker in range(params.attackers_count):
            target_pool = random.sample(
                range(params.defenders_count),
                k=params.target_pool_size)
            attacks_received[min(target_pool)] += focused_attacks
            for target in target_pool:
                attacks_received[target] += spread_attacks

        return self.damage_received(attacks_received, params)

    def simulate_old(self, params, rounds, value_function=None):
        return self.process_rounds(self.simulate_round_old, params, rounds, value_function)

    def simulate_round_old(self, params: Params, value_function=None):
        attacks_received = [0] * params.defenders_count

        for attacker in range(params.attackers_count):
            attacks_received[random.choice(range(params.defenders_count))] += params.attack

        return self.damage_received(attacks_received, params)


def run_simulations(
        rounds,
        ax_range, ax_property,
        x_range, x_property,
        series_range, series_property,
        old_series_range=None, old_series_property=None,
        value_function=None,
        **kwargs):
    s = Simulation()

    axs_data = []
    for ax_value in ax_range:
        dataset = [
            [
                s.simulate(
                    Params.load_base(**{ax_property: ax_value, x_property: x_value, series_property: series_value}),
                    rounds, value_function)
                for series_value in series_range
            ] for x_value in x_range
        ]

        if not old_series_range:
            old_series_range = []

        old_dataset = [
            [
                s.simulate_old(
                    Params.load_base(**{
                        ax_property: ax_value, x_property: x_value,
                        old_series_property: old_series_value,
                        'overwidth_penalty_rate': 2.0
                    }),
                    rounds, value_function)
                for old_series_value in old_series_range
            ] for x_value in x_range
        ]

        axs_data.append([data + old_data for (data, old_data) in zip(dataset, old_dataset)])

    print("Simulation phase done, ran %d simulations with %d rounds " % (s.counter, rounds))
    return axs_data


def display_results(data,
        title,
        ax_range, ax_property,
        x_range, x_property,
        series_labels, y_label='damage dice dealt', **kwargs):

    fig, axs = plt.subplots(len(ax_range))  # Create a figure containing a single axes.
    for i, ax in enumerate(axs):
        ax.set_title(str(ax_range[i]) + ' ' + ax_property)
        ax.set_xticks(x_range)
        ax.set_xlabel(x_property)
        ax.set_ylabel(y_label)
        ax.plot(x_range, data[i], label=series_labels)

        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')

        ax.legend()

    plt.suptitle(title)
    plt.show()


def export_results(data, file_prefix, ax_range, ax_property, x_range, x_property, series_labels, **kwargs):
    for i, ax_value in enumerate(ax_range):
        filename = '../output/%s_%s_%s.csv' % (file_prefix, ax_property, ax_value)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([x_property] + series_labels) # headers
            for row in ([x_value] + data_row for (x_value, data_row) in zip(x_range, data[i])):
                writer.writerow(row)  # headers


def attacker_scenario():
    series_range = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
    return {
        'title': "Average total damage dice dealt by attackers (80 + 26 attack per width) to 20w defenders (700 defense) on a filled frontage across 10000 runs",
        'file_prefix': 'attackers',
        'rounds': 10000,
        'ax_range': (80, 120, 160),
        'ax_property': 'combat_width',
        'x_range': list(range(10, 50)),
        'x_property': 'attacker_width',
        'series_range': series_range,
        'series_property': 'effective_coordination',
        'old_series_range': (0,),
        'old_series_property': '',
        'series_labels': ["Coordination = %d (%d%%)" % ((x-0.35)*100, x*100) for x in series_range] + ['20w pre-NSB rules']
    }


def coordination_scenario():
    series_range = (20, 25, 29, 30, 35, 39, 40, 45)
    return {
        'title': "Average total damage dice dealt by attackers (80 + 26 attack per width) to 20w defenders (700 defense) on a filled frontage across 10000 runs",
        'file_prefix': 'coordination',
        'rounds': 10000,
        'ax_range': (80, 120, 160),
        'ax_property': 'combat_width',
        'x_range': [x / 100 for x in range(35, 70)],
        'x_property': 'effective_coordination',
        'series_range': series_range,
        'series_property': 'attacker_width',
        'old_series_range': None,
        'old_series_property': '',
        'series_labels': ["Attacker width = %d" % x for x in series_range]
    }


def defenders_scenario():
    series_range = (16, 20, 27, 30, 40)
    old_series_range = (20, 40)
    return {
        'title': "Average total damage dice received by defenders (80 + 31 defense per width) by attackers (80 + 26 attack per width, 0.45 effective coordination) on a filled frontage across 10000 runs",
        'file_prefix': 'defenders',
        'rounds': 10000,
        'ax_range': (80, 120, 160),
        'ax_property': 'combat_width',
        'x_range': list(range(2, 50)),
        'x_property': 'defender_width',
        'series_range': series_range,
        'series_property': 'attacker_width',
        'old_series_range': old_series_range,
        'old_series_property': 'attacker_width',
        'series_labels': ["Attacker width = %d" % x for x in series_range] + ["Pre-NSB attacker width = %d" % x for x in old_series_range]
    }

def defenders_org_scenario():
    def value_function(x, params: Params):
        org = round(80 - (140/(2+(params.defender_width/2))))
        return 100 * x / (params.defenders_count * org)
    return dict(defenders_scenario(),
                x_range=list(range(2, 50, 2)),
                file_prefix='defenders_org',
                title="Average normalized ORG damage received by defenders (80 + 31 defense per width) by attackers (80 + 26 attack per width, 0.45 effective coordination) on a filled frontage across 10000 runs",
                y_label='relative ORG damage received',
                value_function=value_function)


def main():
    scenario = defenders_org_scenario()
    data = run_simulations(**scenario)
    export_results(data, **scenario)
    display_results(data, **scenario)


if __name__ == '__main__':
    main()
