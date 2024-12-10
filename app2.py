#(C)Tsubasa Kato at Inspire Search Corp. - Made with ChatGPT (o1 + GPT-4o)
# Version 0.1b - Last Updated on 12/10/2024 20:13PM JST
import os
from flask import Flask, request, jsonify, render_template
import logging
import numpy as np
import py4hw
import time
import mlx.core as mx

logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG
)

app = Flask(__name__)

OPTIMIZATION_RUNS = {}

def log_message(run_id, message):
    if run_id in OPTIMIZATION_RUNS:
        OPTIMIZATION_RUNS[run_id]["log"].append(message)
    logging.debug(message)
    print(message)

# Feature detection for py4hw
has_adder = hasattr(py4hw, 'Adder')
has_multiplier = hasattr(py4hw, 'Multiplier')
has_simulator = hasattr(py4hw, 'Simulator')
has_compute_fitness = hasattr(py4hw, 'compute_fitness')

# Mock classes for fallback if py4hw classes aren't available
class MockCircuit:
    def __init__(self, bitwidth, pipeline_depth, resource_sharing, dsp_usage):
        self.bitwidth = bitwidth
        self.pipeline_depth = pipeline_depth
        self.resource_sharing = resource_sharing
        self.dsp_usage = dsp_usage

    def get_lut_count(self):
        return np.random.randint(100, 200)

    def get_timing_slack(self):
        return float(np.random.uniform(-1.0, 1.0))

    def get_power_estimate(self):
        return float(np.random.uniform(0.1, 5.0))

class MockSimulator:
    def __init__(self, circuit):
        self.circuit = circuit

    def run(self):
        time.sleep(0.01)  # simulate a small delay

def mock_compute_fitness(lut_count, timing_slack, power):
    # A simple heuristic: we want low LUT count and low power, but high timing slack.
    return lut_count * power - timing_slack

def evaluate_design(circuit_type, bitwidth, pipeline_depth, resource_sharing, dsp_usage):
    """
    Evaluate a hardware design using either the real py4hw Adder & Simulator,
    or fallback mock objects if unavailable.
    """
    if circuit_type == 'adder':
        if has_adder:
            circuit = py4hw.Adder(bitwidth, pipeline_depth, resource_sharing, dsp_usage)
        else:
            circuit = MockCircuit(bitwidth, pipeline_depth, resource_sharing, dsp_usage)
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    simulator = py4hw.Simulator(circuit) if has_simulator else MockSimulator(circuit)

    start_time = time.time()
    simulator.run()
    simulation_time = time.time() - start_time

    lut_count = circuit.get_lut_count()
    timing_slack = circuit.get_timing_slack()
    power = circuit.get_power_estimate()

    return {
        'lut_count': lut_count,
        'timing_slack': timing_slack,
        'power': power,
        'simulation_time': simulation_time
    }

def evaluate_population(population, circuit_type, bitwidth):
    """
    Given a population of design parameters, evaluate each design and produce a metrics array.
    """
    # Convert to NumPy for indexing
    np_population = np.array(population)
    n = np_population.shape[0]

    lut_list = []
    timing_list = []
    power_list = []

    for i in range(n):
        pipeline_depth = int(np_population[i, 0])
        resource_sharing_val = int(np_population[i, 1])
        dsp_usage_val = int(np_population[i, 2])

        resource_sharing = (resource_sharing_val == 1)
        dsp_usage = 'low' if dsp_usage_val == 0 else 'high'

        results = evaluate_design(
            circuit_type=circuit_type,
            bitwidth=bitwidth,
            pipeline_depth=pipeline_depth,
            resource_sharing=resource_sharing,
            dsp_usage=dsp_usage
        )

        lut_list.append(results['lut_count'])
        timing_list.append(results['timing_slack'])
        power_list.append(results['power'])

    # Create mx arrays from Python lists
    lut_arr = mx.array(lut_list)
    timing_arr = mx.array(timing_list)
    power_arr = mx.array(power_list)

    metrics_array = mx.stack([lut_arr, timing_arr, power_arr], axis=1)
    return metrics_array

def fitness_function(metrics_array):
    """
    Compute fitness for each design in the population.
    """
    # Convert to NumPy for indexing and scalar extraction
    np_metrics = np.array(metrics_array)
    n = np_metrics.shape[0]
    fitness_list = []

    for i in range(n):
        lut_count = float(np_metrics[i, 0])
        timing_slack = float(np_metrics[i, 1])
        power = float(np_metrics[i, 2])

        if has_compute_fitness:
            fit = py4hw.compute_fitness(lut_count, timing_slack, power)
        else:
            fit = mock_compute_fitness(lut_count, timing_slack, power)
        fitness_list.append(fit)

    return mx.array(fitness_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_optimization():
    try:
        circuit_type = request.form.get('circuit_type', 'adder')
        bitwidth = int(request.form.get('bitwidth', '8'))
        generations = int(request.form.get('generations', '2'))
        population_size = int(request.form.get('population_size', '6'))

        pipeline_depths_np = np.random.choice([0,1,2], size=population_size)
        resource_sharing_np = np.random.choice([0,1], size=population_size)
        dsp_usage_np = np.random.choice([0,1], size=population_size)

        pipeline_depths = mx.array(pipeline_depths_np)
        resource_sharing = mx.array(resource_sharing_np)
        dsp_usage = mx.array(dsp_usage_np)

        population = mx.stack([pipeline_depths, resource_sharing, dsp_usage], axis=1)

        run_id = f"run_{circuit_type}_{bitwidth}"
        OPTIMIZATION_RUNS[run_id] = {
            "circuit_type": circuit_type,
            "bitwidth": bitwidth,
            "generations": generations,
            "population_size": population_size,
            "current_gen": 0,
            "population": population,
            "best_design": None,
            "log": []
        }

        log_message(run_id, f"Started optimization run: {run_id}")
        log_message(run_id, f"Parameters: circuit_type={circuit_type}, bitwidth={bitwidth}, generations={generations}, population_size={population_size}")
        log_message(run_id, f"Initial population: {np.array(population).tolist()}")

        return jsonify({"run_id": run_id})
    except Exception as e:
        logging.error(f"Error in start_optimization: {e}")
        print(f"Error in start_optimization: {e}")
        return jsonify({"error": "Failed to start optimization."}), 500

@app.route('/status', methods=['GET'])
def status():
    """
    Retrieve the current best design and other status information.
    This route converts all mx arrays to numpy before processing to avoid conversion errors.
    """
    try:
        run_id = request.args.get('run_id')
        if run_id not in OPTIMIZATION_RUNS:
            return jsonify({"error": "Invalid run_id"}), 404

        run = OPTIMIZATION_RUNS[run_id]
        population = run["population"]

        # Evaluate population and fitness using the helper functions
        metrics_array = evaluate_population(population, run["circuit_type"], run["bitwidth"])
        fitness_array = fitness_function(metrics_array)

        # Convert to NumPy arrays to safely extract scalars and list forms
        np_metrics = np.array(metrics_array)
        np_fitness = np.array(fitness_array)
        np_population = np.array(population)

        best_idx = np.argmin(np_fitness)

        best_metrics = np_metrics[best_idx].tolist()   # Safe, returns Python list
        best_params = np_population[best_idx].tolist() # Safe, returns Python list

        run["best_design"] = {
            "params": best_params,
            "metrics": best_metrics,
            "fitness": float(np_fitness[best_idx]),  # np_fitness[best_idx] is a NumPy scalar, safe to convert
        }

        current_gen = run["current_gen"]
        log_message(run_id, f"Status requested: Run {run_id} - Gen {current_gen}: Best fitness={np_fitness[best_idx]} with params {best_params}")

        return jsonify({
            "run_id": run_id,
            "circuit_type": run["circuit_type"],
            "bitwidth": run["bitwidth"],
            "current_generation": current_gen,
            "total_generations": run["generations"],
            "best_design": run["best_design"],
            "log": run["log"]
        })
    except Exception as e:
        logging.error(f"Error in status: {e}")
        print(f"Error in status: {e}")
        return jsonify({"error": "Failed to retrieve status."}), 500

@app.route('/next_generation', methods=['POST'])
def next_generation():
    """
    Advance to the next generation by mutating the population parameters.
    This also uses a safe array-to-numpy conversion before indexing.
    """
    try:
        run_id = request.form.get('run_id')
        if run_id not in OPTIMIZATION_RUNS:
            return jsonify({"error": "Invalid run_id"}), 404

        run = OPTIMIZATION_RUNS[run_id]
        if run["current_gen"] >= run["generations"]:
            message = "All generations completed."
            log_message(run_id, message)
            return jsonify({"message": message})

        population = run["population"]

        # Mutate the population
        mutation = mx.random.normal(0, 1, size=population.shape)
        np_population = np.array(population) + np.array(mutation)
        np_population = np.round(np_population)
        np_population[:,0] = np.clip(np_population[:,0], 0, 2)
        np_population[:,1] = np.clip(np_population[:,1], 0, 1)
        np_population[:,2] = np.clip(np_population[:,2], 0, 1)
        population = mx.array(np_population)

        run["population"] = population
        run["current_gen"] += 1

        message = f"Advanced run {run_id} to generation {run['current_gen']}"
        log_message(run_id, message)
        return jsonify({"message": message})
    except Exception as e:
        logging.error(f"Error in next_generation: {e}")
        print(f"Error in next_generation: {e}")
        return jsonify({"error": "Failed to advance generation."}), 500

if __name__ == '__main__':
    logging.debug("Starting Flask app with py4hw and mx integration...")
    print("Starting Flask app with py4hw and mx integration...")
    app.run(host='127.0.0.1', port=3000, debug=True)
