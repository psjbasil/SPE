import random
import numpy as np
import heapq
from excercise1 import MM1Simulator
import matplotlib.pyplot as plt

class TraversalTimeSimulator(MM1Simulator):
    def __init__(self, lambda_, mu, end_time):
        super().__init__(lambda_, mu, end_time)
        self.arrival_times = []     # arrival timestamps for each packet
        self.traversal_times = []   # complete packet delays

    def run(self):
        random.seed(666)
        first_arrival = random.expovariate(self.lambda_)
        heapq.heappush(self.event_list, (first_arrival, 'ARRIVAL'))
        heapq.heappush(self.event_list, (self.end_time, 'END'))

        self.service_queue = []  # used to match arrival and departure for tracking

        total_area = 0.0
        last_time = 0.0
        last_state = 0

        while self.event_list:
            event_time, event_type = heapq.heappop(self.event_list)
            self.clock = event_time
            interval = self.clock - last_time
            total_area += last_state * interval
            if self.clock > 0:
                current_avg = total_area / self.clock
                self.running_time.append(self.clock)
                self.running_avg.append(current_avg)
            last_time = self.clock

            if event_type == 'ARRIVAL':
                self.arrival_times.append(self.clock)
                self.service_queue.append(self.clock)  # store arrival for future match

                next_arrival_time = self.clock + random.expovariate(self.lambda_)
                if next_arrival_time < self.end_time:
                    heapq.heappush(self.event_list, (next_arrival_time, 'ARRIVAL'))

                if self.server_busy:
                    self.queue_length += 1
                else:
                    self.server_busy = True
                    service_time = random.expovariate(self.mu)
                    heapq.heappush(self.event_list, (self.clock + service_time, 'DEPARTURE'))

                self.num_in_system = self.queue_length + (1 if self.server_busy else 0)
                self.time_points.append(self.clock)
                self.state_points.append(self.num_in_system)
                last_state = self.num_in_system

            elif event_type == 'DEPARTURE':
                if self.service_queue:
                    arrival_time = self.service_queue.pop(0)
                    traversal_time = self.clock - arrival_time
                    self.traversal_times.append(traversal_time)

                if self.queue_length > 0:
                    self.queue_length -= 1
                    service_time = random.expovariate(self.mu)
                    heapq.heappush(self.event_list, (self.clock + service_time, 'DEPARTURE'))
                else:
                    self.server_busy = False

                self.num_in_system = self.queue_length + (1 if self.server_busy else 0)
                self.time_points.append(self.clock)
                self.state_points.append(self.num_in_system)
                last_state = self.num_in_system

            elif event_type == 'END':
                break

        total_area += last_state * (self.end_time - last_time)
        self.time_average = total_area / self.end_time


def control_variate_estimator(sim: TraversalTimeSimulator, mu, lambda_):
    # sample mean of traversal times
    Y = np.array(sim.traversal_times)
    Y_bar = np.mean(Y)

    # control variate: average packets in system (already estimated)
    X_bar = sim.time_average
    X_theory = lambda_ / (mu - lambda_)

    # estimate covariance and variance
    cov = np.cov(Y, [sim.state_points[:len(Y)]])[0, 1]
    var_X = np.var(sim.state_points[:len(Y)])
    c_opt = -cov / var_X if var_X != 0 else 0

    # control variate estimator
    Y_cv = Y_bar + c_opt * (X_bar - X_theory)
    return Y_cv, Y_bar, np.var(Y), c_opt


def run_experiment(lambda_, mu, end_time):
    sim = TraversalTimeSimulator(lambda_, mu, end_time)
    sim.run()
    Y_cv, Y_naive, var_naive, c_opt = control_variate_estimator(sim, mu, lambda_)

    print(f"\nρ = {lambda_ / mu:.2f}")
    print(f"Naive mean = {Y_naive:.5f}, Var = {var_naive:.5f}")
    print(f"Control variate mean = {Y_cv:.5f}, c* = {c_opt:.4f}")
    print(f"Expected theoretical mean = {1 / (mu - lambda_):.5f}")
    return Y_naive, Y_cv, sim


def plot_traversal_comparison(results):
    rhos = [lambda_/mu for (lambda_, mu) in results.keys()]
    labels = [f"ρ={rho:.2f}" for rho in rhos]
    naive_vals = [v[0] for v in results.values()]
    cv_vals = [v[1] for v in results.values()]
    theo_vals = [1 / (mu - lambda_) for (lambda_, mu) in results.keys()]

    x = np.arange(len(labels))
    width = 0.3
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, naive_vals, width, label='Naive')
    plt.bar(x, cv_vals, width, label='Control Variate')
    plt.plot(x, theo_vals, 'r--', label='Theoretical', linewidth=2)

    plt.xticks(x, labels)
    plt.ylabel('Average Traversal Time')
    plt.title('Traversal Time Estimation Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparison_traversal_time.png")
    plt.close()

if __name__ == "__main__":
    end_time = 10000
    param_sets = [
        (0.6, 2.0),  # ρ=0.3
        (1.0, 2.0),  # ρ=0.5
        (1.4, 2.0),  # ρ=0.7
        (1.8, 2.0),  # ρ=0.9
    ]

    results = {}
    for lambda_, mu in param_sets:
        Y_naive, Y_cv, sim = run_experiment(lambda_, mu, end_time)
        results[(lambda_, mu)] = (Y_naive, Y_cv)

    plot_traversal_comparison(results)
