import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np  

# Constants
constants = {
    "time_periods": 2,
    "energy_sources": ["Grid", "Solar"],
    "scenarios": ["S_high", "S_avg", "S_low"],
    #####CHANGING THESE VALUES WHEN SOLVING FOR EACH SCENARIO INDIVIDUALLY#####
    "probs": {"S_high": 0.3, "S_avg": 0.5, "S_low": 0.2}
}

# Input Data
data = {
    'Cost_grid': [1300, 700],
    'Cost_solar': 0,  # Solar has zero marginal cost
    'Cost_battery': 0,
    'aFRR_price': 1275,  # aFRR price 
    'Energy_demand': [240, 120],  # Constant demand for simplicity
    
    'Solar_generation': {  # Solar generation also varies
        'S_high': [24,0],  # High solar generation
        'S_avg': [14.4,0],  # Average solar generation
        'S_low': [6,0],  # Low solar generation
    },
    'Battery_charge_eff': 1.1,
    'Battery_discharge_eff': 0.9,
    'Max_battery_capacity': 60,
    'Max_battery_charge_discharge': 12,
    #'Max_battery_discharge': 12,
    'Grid_capacity': 300,
    'Initial_battery_storage': 12
}


#Mathematical formulation first stage
def Obj_first_stage(m):
        return -sum(m.x_aFRR[t] * m.P_aFRR[t] for t in m.T) + m.alpha
def ReserveMarketLimit_first_stage(m, t):
        return m.x_aFRR[t] <= m.P_max #max battery discharge
def Optimality_cut(m, c):           # Create Benders Optimality cuts from first stage
        return m.alpha >= m.Phi[c] + sum(m.Lambda[c, t] * (m.x_aFRR[t] - m.x_hat[c, t])\
        for t in m.T)


#Mathematical formulation second stage
def Obj_second_stage(m):
    return sum(m.pi[s] * sum(
        m.y_supply[t, s, 'Grid'] * m.C_grid[t] + \
        m.y_supply[t, s, 'Solar'] * m.C_solar - \
        m.z_export[t, s] * m.C_exp[t] + \
        m.penalty * m.slack_energy_balance[t, s]  # Penalize artificial variable usage
        for t in m.T) for s in m.S)
def EnergyBalance_sec(m, t, s):
        return m.D[t] + m.x_aFRR[t] == sum(m.y_supply[t, s, i] for i in m.I) -\
        m.z_export[t, s] + m.q_discharge[t, s] - m.eta_charge * m.q_charge[t, s] +\
        m.slack_energy_balance[t, s]
def ReserveMarketLimit_sec(m, t, s):
        return m.x_aFRR[t] <=  m.q_discharge[t,s] + m.slack_energy_balance[t, s] 
def StorageDynamics(m, t, s):
        if t == 1:
            return m.e_storage[t, s] == m.I_INIT + m.q_charge[t, s] -\
            m.q_discharge[t, s] / m.eta_discharge 
        else:
            return m.e_storage[t, s] == m.e_storage[t-1, s] + m.q_charge[t, s] -\
            m.q_discharge[t, s] / m.eta_discharge 
        
def BatteryLimits(m, t, s):
    return m.e_storage[t, s] <= m.E_max
def ChargeLimit(m, t, s):
    return m.q_charge[t, s] + m.q_discharge[t, s]/m.eta_discharge <= m.P_max
#def DischargeLimit(m, t, s):
#    return m.q_discharge[t, s] <= m.P_discharge_max
def EnsureCapacityForaFRR(m, t, s):
        if t == 1:
            return m.I_INIT >= m.x_aFRR[t]
        else:
            return m.e_storage[t-1, s] - m.x_aFRR[t] >= 0 
        
def ImportLimit(m, t, s):
    return m.y_supply[t, s, 'Grid'] <= m.G_max
def SolarPowerLimit(m, t, s):
    return m.y_supply[t, s, 'Solar'] == m.G_solar[t, s]
def ExportLimit_sec(m, t, s):
    return m.z_export[t, s] + m.x_aFRR[t] <= m.G_max + m.slack_energy_balance[t, s]
def Lambda_constraint(m, t):
    return m.x_aFRR[t] == m.x_hat[t]
    


#Master problem: Define the model setup
def First_stage_model(data, constants, Cuts):
    
    m = pyo.ConcreteModel()
    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.C = pyo.Set(initialize=Cuts["Set"])  # Set for cuts 

    # Parameters
    m.P_aFRR = pyo.Param(m.T, initialize=data['aFRR_price'])  # aFRR price
    m.G_max = pyo.Param(initialize=data['Grid_capacity'])  # Grid capacity limit
    m.P_max = pyo.Param(initialize=data['Max_battery_charge_discharge'])


    # Parameters for cuts
    m.Phi = pyo.Param(m.C, initialize=Cuts["Phi"], default = 0)  # Initialize cuts
    m.Lambda = pyo.Param(m.C, m.T, initialize=Cuts["lambda"])  
    m.x_hat = pyo.Param(m.C, m.T, initialize=Cuts["x_hat"]) 

    # Variables
    m.x_aFRR = pyo.Var(m.T, within=pyo.NonNegativeReals)  # aFRR reserve (first-stage decision)
    m.alpha = pyo.Var(bounds=(0, 1000000))  # Approximates second-stage cost

    # Objective: minimize first-stage cost with the approximation of the second-stage cost (alpha)
    m.obj = pyo.Objective(rule=Obj_first_stage, sense=pyo.minimize)
    # Constraints
    m.ReserveMarketLimit_first_stage = pyo.Constraint(m.T, rule=ReserveMarketLimit_first_stage)
    m.Optimality_cut = pyo.Constraint(m.C, rule=Optimality_cut)
    
    return m



#Subproblem: Define model setup
def Second_stage_model(data, constants, x_hat):
    
    m = pyo.ConcreteModel()

    # Sets
    m.T = pyo.RangeSet(1, constants["time_periods"])  # Time periods
    m.I = pyo.Set(initialize=constants["energy_sources"])  # Energy sources
    m.S = pyo.Set(initialize=constants["scenarios"])  # Scenarios


    # Parameters
    m.C_grid = pyo.Param(m.T, initialize={t: data['Cost_grid'][t-1] for t in m.T})
    m.C_solar = pyo.Param(initialize=data['Cost_solar'])  # Solar cost is constant
    m.C_battery = pyo.Param(initialize=data['Cost_battery'])  # Battery cost is constant
    m.penalty = 10000  # Large penalty for using the artificial variables
    m.C_exp = pyo.Param(m.T, initialize={t: 0.9 * data['Cost_grid'][t-1] for t in m.T})
    m.P_aFRR = pyo.Param(m.T, initialize=data['aFRR_price'])  # aFRR price
    
    m.D = pyo.Param(m.T, initialize={t: data['Energy_demand'][t-1] 
    for t in range(1, constants["time_periods"] + 1)})
    
    m.G_solar = pyo.Param(m.T, m.S, initialize={(t, s): data['Solar_generation'][s][t-1] 
    for t in range(1, constants["time_periods"] + 1) for s in constants["scenarios"]})
    
    m.eta_charge = pyo.Param(initialize=data['Battery_charge_eff'])
    m.eta_discharge = pyo.Param(initialize=data['Battery_discharge_eff'])
    m.E_max = pyo.Param(initialize=data['Max_battery_capacity'])
    m.P_max = pyo.Param(initialize=data['Max_battery_charge_discharge'])
    #m.P_discharge_max = pyo.Param(initialize=data['Max_battery_discharge'])
    m.G_max = pyo.Param(initialize=data['Grid_capacity'])
    m.I_INIT = pyo.Param(initialize=data['Initial_battery_storage'])
    m.pi = pyo.Param(m.S, initialize=constants["probs"])  # Probability of each scenario
    #Parameter for cuts
    
    m.x_hat = pyo.Param(m.T, initialize = x_hat)
    # Variables 
    m.x_aFRR = pyo.Var(m.T, within=pyo.NonNegativeReals)  # aFRR reserve (first-stage decision)
    # Energy supply from sources with bounds
    m.y_supply = pyo.Var(m.T, m.S, m.I, within=pyo.NonNegativeReals, bounds=(0, m.G_max)) 
    m.z_export = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Energy exported to the grid
    m.q_charge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery charge
    m.q_discharge = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)  # Battery discharge
    # Battery energy storage (shared across scenarios)
    m.e_storage = pyo.Var(m.T,m.S, within=pyo.NonNegativeReals, bounds=(0, m.E_max))
    # Add artificial variables to the model to account for infeasibilities
    m.slack_energy_balance = pyo.Var(m.T, m.S, within=pyo.NonNegativeReals)

    # Objective function for the second stage problem
    m.obj = pyo.Objective(rule=Obj_second_stage, sense=pyo.minimize)

    # Constraints
    m.EnergyBalance_sec = pyo.Constraint(m.T, m.S, rule=EnergyBalance_sec)
    m.ReserveMarketLimit_sec = pyo.Constraint(m.T,m.S, rule=ReserveMarketLimit_sec)
    m.StorageDynamics = pyo.Constraint(m.T, m.S, rule=StorageDynamics)
    m.BatteryLimits = pyo.Constraint(m.T, m.S, rule=BatteryLimits)
    m.ChargeLimit = pyo.Constraint(m.T, m.S, rule=ChargeLimit)
    #m.DischargeLimit = pyo.Constraint(m.T, m.S, rule=DischargeLimit)
    m.EnsureCapacityForaFRR = pyo.Constraint(m.T, m.S, rule=EnsureCapacityForaFRR)
    #m.BatterySupplyLimit = pyo.Constraint(m.T, m.S, rule=BatterySupplyLimit)
    m.ImportLimit = pyo.Constraint(m.T, m.S, rule=ImportLimit)
    m.SolarPowerLimit = pyo.Constraint(m.T, m.S, rule=SolarPowerLimit)
    m.ExportLimit_sec = pyo.Constraint(m.T, m.S, rule=ExportLimit_sec)
    m.Lambda_constraint = pyo.Constraint(m.T, rule=Lambda_constraint)

    return m

# Function for creating new linear cuts for optimization problem (Inspired by code given as example for Benders decomposition in class)
def Create_cuts(Cuts, m):
    cut_it = len(Cuts["Set"])  # Find the current cut iteration index
    Cuts["Set"].append(cut_it)  # Add a new cut to the set

    # Add new Phi value for the new cut
    Cuts["Phi"][cut_it] = pyo.value(m.obj)

    # Add new lambda and x_hat values for the new cut
    for t in m.T:
        Cuts["lambda"][cut_it,t] = m.dual[m.Lambda_constraint[t]]
        Cuts["x_hat"][cut_it,t] = pyo.value(m.x_hat[t])

    return(Cuts)

# Solve the model
def SolveModel(m): 
    solver = SolverFactory('gurobi')
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = solver.solve(m, tee=True)
    
    return results, m


#New function for doing Benders decomposition
def Benders_decomposition(data, constants, Cuts):

    #Setup for Benders decomposition - Perform this for x-iterations
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    import time
    initial_time = time.time()

    #Using a for loop for iteration
    for i in range(10):

        #Solve 1st stage problem
        m_first_stage = First_stage_model(data, constants, Cuts)
        SolveModel(m_first_stage)
    
        #First stage result process with x_hat value using numerical indices
        x_hat = {t: pyo.value(m_first_stage.x_aFRR[t]) \
        for t in range(1, constants["time_periods"] + 1)}

    
        
        #Printing first stage results
        print(f"Iteration {i}")
        for t in x_hat:
            print(f"t{t}: {x_hat[t]}")
        #input()

        #Setup and solve 2nd stage problem
        m_second_stage = Second_stage_model(data, constants, x_hat)
        SolveModel(m_second_stage)

        # Print the slack variables for each time period and scenario
        for t in m_second_stage.T:
            x_aFRR = pyo.value(m_second_stage.x_aFRR[t])
            print(f"Iteration {i}, Time {t} x_aFRR: {x_aFRR}")
            for s in m_second_stage.S:
                slack_value = pyo.value(m_second_stage.slack_energy_balance[t,s])
                print(f"Iteration {i}, Time {t} Slack: {slack_value}")

        #Creating cuts for the first stage model
        Cuts = Create_cuts(Cuts,m_second_stage)
        
        #Print results for second stage
        print("UB:",pyo.value(m_first_stage.alpha.value),"- LB:",pyo.value(m_second_stage.obj))
        print("Objective value of problem:", pyo.value(m_second_stage.obj()-\
        m_first_stage.x_aFRR[1].value*1275-m_first_stage.x_aFRR[2]*1275)) 
        print("Cut information acquired:")
        for component in Cuts:
            if component == "lambda" or component == "x_hat":
                for t in m_second_stage.T:
                    # Check if t exists in Cuts[component]
                    if t in Cuts[component]:
                        print(component, t, Cuts[component][t])
                    
            else:
                print(component, Cuts[component])

        #input()

        #Performing a convergence check with upper and lower bound
        print("UB:",pyo.value(m_first_stage.alpha.value),"- LB:",pyo.value(m_second_stage.obj))

        #input()
        print(time.time()-initial_time)
        return()
    
import pandas as pd

# Function to export results to Excel after Stochastic Dynamic Programming
import pandas as pd

# Function to export results to Excel after Stochastic Dynamic Programming
def Stochastic_Dynamic_Programming():
    # Initial setup for SDP
    Minimum = 0
    Maximum = 12  # Max battery discharge capacity
    num_points = 100
    List_of_jumps = np.linspace(Minimum, Maximum, num_points).tolist()

    # Initial values for the decision variable in the first stage
    x_aFRR_initial_values = List_of_jumps

    # Initialize cuts
    Cuts = {}
    Cuts["Set"] = []
    Cuts["Phi"] = {}
    Cuts["lambda"] = {}
    Cuts["x_hat"] = {}

    # Collect results to export to Excel
    results = {
        "Time Period": [],
        "Scenario": [],        # Added "Scenario" key
        "x_hat": [],
        "z_export": [],
        "y_supply": [],
        "q_charge": [],
        "q_discharge": [],
        "e_storage": []
    }

    for initial_value in x_aFRR_initial_values:
        # Define x_hat as a dictionary indexed by time periods
        X_hat = {t: initial_value for t in range(1, constants["time_periods"] + 1)}
        
        # If combination is valid, solve the second stage problem
        if all(X_hat[t] <= Maximum for t in X_hat):
            # Setup and solve the second-stage problem
            m_second_stage = Second_stage_model(data, constants, X_hat)
            SolveModel(m_second_stage)

            # Create cuts for the first stage model
            Cuts = Create_cuts(Cuts, m_second_stage)
        else:
            continue

    # Solve the first stage with the created cuts
    m_first_stage = First_stage_model(data, constants, Cuts)
    SolveModel(m_first_stage)

    # Get the value of x_hat from the first-stage solution
    X_hat = {t: pyo.value(m_first_stage.x_aFRR[t]) for t in range(1, constants["time_periods"] + 1)}

   # Store the results for all scenarios in the dictionary
    for t in X_hat:
        for scenario in constants["scenarios"]:
            results["Time Period"].append(t)
            results["Scenario"].append(scenario)  # Storing scenario here
            results["x_hat"].append(X_hat[t])
            results["z_export"].append(pyo.value(m_second_stage.z_export[t, scenario]))
            results["y_supply"].append(pyo.value(m_second_stage.y_supply[t, scenario, "Grid"]))
            results["q_charge"].append(pyo.value(m_second_stage.q_charge[t, scenario]))
            results["q_discharge"].append(pyo.value(m_second_stage.q_discharge[t, scenario]))
            results["e_storage"].append(pyo.value(m_second_stage.e_storage[t, scenario]))
    
    # Create a DataFrame from the results
    df_results = pd.DataFrame(results)

    # Export the DataFrame to Excel
    excel_filename = "SDP_results.xlsx"
    df_results.to_excel(excel_filename, index=False)

    print(f"Results have been saved to {excel_filename}")
    return df_results

# Calling the function to execute and export results
Stochastic_Dynamic_Programming()