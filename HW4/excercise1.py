import heapq
import random
import matplotlib.pyplot as plt

class MM1Simulator:
    def __init__(self, lambda_, mu, end_time):
        self.lambda_ = lambda_  # arrival rate
        self.mu = mu            # service rate
        self.end_time = end_time # simulation duration
        self.clock = 0.0         # current simulation time
        self.server_busy = False # server status
        self.queue_length = 0    # number of packets in the queue
        self.num_in_system = 0  # total packets in system (queue + server)
        self.event_list = []     # priority queue for events
        # state recording: (time, num_in_system)
        self.time_points = [0.0]  # times of state changes
        self.state_points = [0]   # state after each change
        # for running average
        self.running_time = []    # times for running average
        self.running_avg = []     # running average values

    def run(self):
        # schedule first arrival and end event
        random.seed(666)  # set seed for reproducibility
        first_arrival = random.expovariate(self.lambda_)
        heapq.heappush(self.event_list, (first_arrival, 'ARRIVAL'))
        heapq.heappush(self.event_list, (self.end_time, 'END'))
        
        # initialize running average tracking
        total_area = 0.0
        last_time = 0.0
        last_state = 0
        
        while self.event_list:
            event_time, event_type = heapq.heappop(self.event_list)
            self.clock = event_time
            
            # record state for running average up to this event
            interval = self.clock - last_time
            total_area += last_state * interval
            if self.clock > 0:
                current_avg = total_area / self.clock
                self.running_time.append(self.clock)
                self.running_avg.append(current_avg)
            last_time = self.clock
            
            if event_type == 'ARRIVAL':
                # schedule next arrival if within simulation time
                next_arrival_time = self.clock + random.expovariate(self.lambda_)
                if next_arrival_time < self.end_time:
                    heapq.heappush(self.event_list, (next_arrival_time, 'ARRIVAL'))
                
                # process current arrival
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
        
        # final segment for total average
        total_area += last_state * (self.end_time - last_time)
        self.time_average = total_area / self.end_time

def plot_system_packets(simulators, end_segment=50):
    """
    plot system packet count changes for multiple simulators
    simulators: list of tuples containing (simulator instance, label)
    """
    plt.figure(figsize=(12, 6))
    for sim, label in simulators:
        segment_times = [t for t in sim.time_points if t <= end_segment]
        segment_states = sim.state_points[:len(segment_times)]
        plt.step(segment_times, segment_states, where='post', label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Packets in System')
    plt.title('Number of Packets in System over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('system_packets.png')
    plt.close()

def plot_running_average(simulators):
    """
    plot running average comparison with theoretical values for multiple simulators
    simulators: list of tuples containing (simulator instance, theoretical value, rho value, label)
    """
    plt.figure(figsize=(12, 6))
    for sim, theoretical, rho, label in simulators:
        plt.plot(sim.running_time, sim.running_avg, 
                label=f'Simulation (ρ={rho:.2f})')
        plt.axhline(y=theoretical, color='r', linestyle='--', 
                   label=f'Theoretical (ρ={rho:.2f}): {theoretical:.2f}')
    
    plt.xlabel('Time')
    plt.ylabel('Running Average Packets')
    plt.title('Running Average of Packets in System')
    plt.legend()
    plt.grid(True)
    plt.savefig('running_average.png')
    plt.close()

def run_simulation(lambda_, mu, end_time):
    """
    run a single simulation and return results
    """
    rho = lambda_ / mu
    theoretical = rho / (1 - rho)
    sim = MM1Simulator(lambda_, mu, end_time)
    sim.run()
    return sim, theoretical, rho

if __name__ == "__main__":
    # simulation parameters
    end_time = 10000
    simulations = [
        (0.6, 2.0),  # ρ = 0.3
        (1.0, 2.0),  # ρ = 0.5
        (1.4, 2.0),  # ρ = 0.7
        (1.8, 2.0)   # ρ = 0.9
    ]
    
    # run all simulations
    results = []
    for lambda_, mu in simulations:
        sim, theoretical, rho = run_simulation(lambda_, mu, end_time)
        results.append((sim, theoretical, rho))
        print(f"ρ={rho:.2f}: Simulation Average={sim.time_average:.4f}, Theoretical={theoretical:.4f}")
    
    # plot system packet count changes
    plot_system_packets([(r[0], f'ρ={r[2]:.2f}') for r in results])
    
    # plot running average comparison
    plot_running_average([(r[0], r[1], r[2], f'ρ={r[2]:.2f}') for r in results])