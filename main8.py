import json
import pandas as pd

# Cargar el JSON desde el archivo
with open('D:\\FRANZ\\RL\\ejemplo.json', 'r') as f:
    grid = json.load(f)

# Parámetros
gamma = 0.8  # factor de descuento
threshold = 0.01  # umbral de convergencia
actions = ["N", "S", "E", "W"]
standard_cost = 1  # Costo estándar para transiciones normales
deadend_cost = 1  # Costo alto para estados deadend

# Inicialización
policy = {state: {action: 1/len(actions) for action in actions} for state in grid}
values = {state: 0 for state in grid}

# Evaluación de la política
def policy_evaluation(policy, grid, values, gamma, threshold):
    while True:
        delta = 0
        for state in grid:
            v = values[state]
            new_value = 0
            for action in actions:
                action_prob = policy[state][action]
                for adj in grid[state]["Adj"]:
                    next_state = adj["name"]
                    prob = adj["A"].get(action, 0)
                    if grid[next_state]["goal"]:
                        cost = 0  # Costo cero para el estado objetivo
                    elif grid[next_state]["deadend"]:
                        cost = deadend_cost
                    else:
                        cost = standard_cost
                    new_value += action_prob * prob * (cost + gamma * values[next_state])
            values[state] = new_value
            delta = max(delta, abs(v - new_value))
        if delta < threshold:
            break

# Mejora de la política
def policy_improvement(policy, grid, values, gamma):
    policy_stable = True
    for state in grid:
        old_action = max(policy[state], key=policy[state].get)
        action_values = {}
        for action in actions:
            action_values[action] = 0
            for adj in grid[state]["Adj"]:
                next_state = adj["name"]
                prob = adj["A"].get(action, 0)
                if grid[next_state]["goal"]:
                    cost = 0  # Costo cero para el estado objetivo
                elif grid[next_state]["deadend"]:
                    cost = deadend_cost
                else:
                    cost = standard_cost
                action_values[action] += prob * (cost + gamma * values[next_state])
        best_action = min(action_values, key=action_values.get)
        policy[state] = {a: 1 if a == best_action else 0 for a in actions}
        if old_action != best_action:
            policy_stable = False
    return policy_stable

# Iteración de la política
def policy_iteration(grid, policy, values, gamma, threshold):
    while True:
        policy_evaluation(policy, grid, values, gamma, threshold)
        if policy_improvement(policy, grid, values, gamma):
            break
    return policy, values

# Iteración de valor
def value_iteration(grid, values, gamma, threshold):
    while True:
        delta = 0
        for state in grid:
            v = values[state]
            action_values = {}
            for action in actions:
                action_values[action] = 0
                for adj in grid[state]["Adj"]:
                    next_state = adj["name"]
                    prob = adj["A"].get(action, 0)
                    if grid[next_state]["goal"]:
                        cost = 0  # Costo cero para el estado objetivo
                    elif grid[next_state]["deadend"]:
                        cost = deadend_cost
                    else:
                        cost = standard_cost
                    action_values[action] += prob * (cost + gamma * values[next_state])
            values[state] = min(action_values.values())
            delta = max(delta, abs(v - values[state]))
        if delta < threshold:
            break
    # Derivar la política óptima a partir de los valores
    policy = {}
    for state in grid:
        action_values = {}
        for action in actions:
            action_values[action] = 0
            for adj in grid[state]["Adj"]:
                next_state = adj["name"]
                prob = adj["A"].get(action, 0)
                if grid[next_state]["goal"]:
                    cost = 0  # Costo cero para el estado objetivo
                elif grid[next_state]["deadend"]:
                    cost = deadend_cost
                else:
                    cost = standard_cost
                action_values[action] += prob * (cost + gamma * values[next_state])
        best_action = min(action_values, key=action_values.get)
        policy[state] = {a: 1 if a == best_action else 0 for a in actions}
    return policy, values

# Ejecutar la iteración de la política
optimal_policy, optimal_values = policy_iteration(grid, policy, values, gamma, threshold)

# Determinar el estado de destino según la política óptima
def determine_destination(state, policy, grid):
    best_action = max(policy[state], key=policy[state].get)
    for adj in grid[state]["Adj"]:
        if best_action in adj["A"]:
            return adj["name"]
    return None

# Crear un DataFrame para mostrar la política y el estado de destino
policy_df = pd.DataFrame(optimal_policy).T
policy_df.columns = ['North', 'South', 'East', 'West']
policy_df.index.name = 'State'

# Agregar columna para el estado de destino
policy_df['Best_Action'] = policy_df.idxmax(axis=1)
#policy_df['Destination'] = policy_df.index.map(lambda state: determine_destination(state, optimal_policy, grid))

# Crear un DataFrame para mostrar los valores
values_df = pd.DataFrame(optimal_values.items(), columns=['State', 'Value'])

# Unir ambos DataFrames
result_df = policy_df.join(values_df.set_index('State'))

# Mostrar los DataFrames
print("Policy Iteration Results:")
print(result_df)

# Ejecutar la iteración de valor
value_policy, value_values = value_iteration(grid, values, gamma, threshold)

# Crear un DataFrame para mostrar la política y el estado de destino
value_policy_df = pd.DataFrame(value_policy).T
value_policy_df.columns = ['North', 'South', 'East', 'West']
value_policy_df.index.name = 'State'

# Agregar columna para el estado de destino
value_policy_df['Best_Action'] = value_policy_df.idxmax(axis=1)
#value_policy_df['Destination'] = value_policy_df.index.map(lambda state: determine_destination(state, value_policy, grid))

# Crear un DataFrame para mostrar los valores
value_values_df = pd.DataFrame(value_values.items(), columns=['State', 'Value'])

# Unir ambos DataFrames
value_result_df = value_policy_df.join(value_values_df.set_index('State'))

# Mostrar los DataFrames
print("\nValue Iteration Results:")
print(value_result_df)
