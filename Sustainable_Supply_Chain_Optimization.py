import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
n = 3  # number of products
p = [100, 80, 60]  # selling prices ($/unit)
c = [50, 40, 30]  # production costs ($/unit)
profit = [p[i] - c[i] for i in range(n)]  # net profit per unit
e = [10, 5, 2]  # emissions (kg CO2/unit)
l = [1, 2, 3]  # labor (hours/unit)
K = 1000  # production capacity (units)
d = [500, 600, 700]  # demand (units)
B = 30000  # budget ($)
E = 3000  # emission cap (kg CO2)
L_min = 1000  # minimum labor (hours)

# Helper function to print results
def print_results(model, x, method_name):
    if model.status == GRB.OPTIMAL:
        print(f"\n{method_name} Results:")
        print(f"Objective Value: {model.objVal:.2f}")
        for i in range(n):
            print(f"Product {i+1} production: {x[i].X:.2f} units")
        print(f"Profit: {sum(profit[i] * x[i].X for i in range(n)):.2f}")
        print(f"Emissions: {sum(e[i] * x[i].X for i in range(n)):.2f}")
        print(f"Labor: {sum(l[i] * x[i].X for i in range(n)):.2f}")
    else:
        print(f"{method_name}: No optimal solution found.")

# 1. Aggregated Method
model_agg = gp.Model("Aggregated")
x_agg = model_agg.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
w1, w2, w3 = 1, 0.01, 0.001  # arbitrary weights
obj_agg = gp.quicksum(profit[i] * x_agg[i] for i in range(n)) - w2 * gp.quicksum(e[i] * x_agg[i] for i in range(n)) + w3 * gp.quicksum(l[i] * x_agg[i] for i in range(n))
model_agg.setObjective(obj_agg, GRB.MAXIMIZE)
model_agg.addConstr(gp.quicksum(x_agg[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_agg.addConstr(x_agg[i] <= d[i], f"demand_{i}")
model_agg.addConstr(gp.quicksum(c[i] * x_agg[i] for i in range(n)) <= B, "budget")
model_agg.addConstr(gp.quicksum(e[i] * x_agg[i] for i in range(n)) <= E, "emissions")
model_agg.addConstr(gp.quicksum(l[i] * x_agg[i] for i in range(n)) >= L_min, "labor")
model_agg.optimize()
print_results(model_agg, x_agg, "Aggregated Method")

# 2. Weighted Method (with normalization)
# Find ideal points
model_profit = gp.Model("Max_Profit")
x_p = model_profit.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
model_profit.setObjective(gp.quicksum(profit[i] * x_p[i] for i in range(n)), GRB.MAXIMIZE)
model_profit.addConstr(gp.quicksum(x_p[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_profit.addConstr(x_p[i] <= d[i], f"demand_{i}")
model_profit.addConstr(gp.quicksum(c[i] * x_p[i] for i in range(n)) <= B, "budget")
model_profit.optimize()
if model_profit.status == GRB.OPTIMAL:
    max_profit = model_profit.objVal
else:
    max_profit = 1  # avoid division by zero

model_emissions = gp.Model("Min_Emissions")
x_e = model_emissions.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
model_emissions.setObjective(gp.quicksum(e[i] * x_e[i] for i in range(n)), GRB.MINIMIZE)
model_emissions.addConstr(gp.quicksum(x_e[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_emissions.addConstr(x_e[i] <= d[i], f"demand_{i}")
model_emissions.addConstr(gp.quicksum(c[i] * x_e[i] for i in range(n)) <= B, "budget")
model_emissions.addConstr(gp.quicksum(l[i] * x_e[i] for i in range(n)) >= L_min, "labor")
model_emissions.optimize()
if model_emissions.status == GRB.OPTIMAL:
    min_emissions = model_emissions.objVal
else:
    min_emissions = 1

model_labor = gp.Model("Max_Labor")
x_l = model_labor.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
model_labor.setObjective(gp.quicksum(l[i] * x_l[i] for i in range(n)), GRB.MAXIMIZE)
model_labor.addConstr(gp.quicksum(x_l[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_labor.addConstr(x_l[i] <= d[i], f"demand_{i}")
model_labor.addConstr(gp.quicksum(c[i] * x_l[i] for i in range(n)) <= B, "budget")
model_labor.addConstr(gp.quicksum(e[i] * x_l[i] for i in range(n)) <= E, "emissions")
model_labor.optimize()
if model_labor.status == GRB.OPTIMAL:
    max_labor = model_labor.objVal
else:
    max_labor = 1

# Weighted method
model_weighted = gp.Model("Weighted")
x_w = model_weighted.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
obj_w = (gp.quicksum(profit[i] * x_w[i] for i in range(n)) / max_profit) - \
        (gp.quicksum(e[i] * x_w[i] for i in range(n)) / min_emissions) + \
        (gp.quicksum(l[i] * x_w[i] for i in range(n)) / max_labor)
model_weighted.setObjective(obj_w, GRB.MAXIMIZE)
model_weighted.addConstr(gp.quicksum(x_w[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_weighted.addConstr(x_w[i] <= d[i], f"demand_{i}")
model_weighted.addConstr(gp.quicksum(c[i] * x_w[i] for i in range(n)) <= B, "budget")
model_weighted.addConstr(gp.quicksum(e[i] * x_w[i] for i in range(n)) <= E, "emissions")
model_weighted.addConstr(gp.quicksum(l[i] * x_w[i] for i in range(n)) >= L_min, "labor")
model_weighted.optimize()
print_results(model_weighted, x_w, "Weighted Method")

# 3. ε-Constrained Method
model_epsilon = gp.Model("Epsilon_Constrained")
x_ec = model_epsilon.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
model_epsilon.setObjective(gp.quicksum(profit[i] * x_ec[i] for i in range(n)), GRB.MAXIMIZE)
model_epsilon.addConstr(gp.quicksum(x_ec[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_epsilon.addConstr(x_ec[i] <= d[i], f"demand_{i}")
model_epsilon.addConstr(gp.quicksum(c[i] * x_ec[i] for i in range(n)) <= B, "budget")
model_epsilon.addConstr(gp.quicksum(e[i] * x_ec[i] for i in range(n)) <= 2500, "emissions_cap")
model_epsilon.addConstr(gp.quicksum(l[i] * x_ec[i] for i in range(n)) >= 1300, "labor_min")
model_epsilon.optimize()
print_results(model_epsilon, x_ec, "ε-Constrained Method")

# 4. Lexicographic Method
model_lex = gp.Model("Lexicographic")
x_lex = model_lex.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
model_lex.setObjectiveN(gp.quicksum(profit[i] * x_lex[i] for i in range(n)), index=0, priority=1, name="profit")
#model_lex.setObjectiveN(gp.quicksum(e[i] * x_lex[i] for i in range(n)), index=1, priority=2, name="emissions", sense=GRB.MINIMIZE)
# Minimize emissions by negating them (Gurobi uses a single model sense)
model_lex.setObjectiveN(-gp.quicksum(e[i] * x_lex[i] for i in range(n)), index=1, priority=2, name="neg_emissions")
model_lex.setObjectiveN(gp.quicksum(l[i] * x_lex[i] for i in range(n)), index=2, priority=3, name="labor")
model_lex.addConstr(gp.quicksum(x_lex[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_lex.addConstr(x_lex[i] <= d[i], f"demand_{i}")
model_lex.addConstr(gp.quicksum(c[i] * x_lex[i] for i in range(n)) <= B, "budget")
model_lex.optimize()
if model_lex.status == GRB.OPTIMAL:
    print("\nLexicographic Method Results:")
    print(f"Profit: {model_lex.getObjective(0).getValue():.2f}")
    print(f"Emissions: {-model_lex.getObjective(1).getValue():.2f}")
    print(f"Labor: {model_lex.getObjective(2).getValue():.2f}")
    for i in range(n):
        print(f"Product {i+1} production: {x_lex[i].X:.2f} units")
else:
    print("Lexicographic Method: No optimal solution found.")

# 5. Goal Programming
model_goal = gp.Model("Goal_Programming")
x_g = model_goal.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
d1p = model_goal.addVar(lb=0, name="d1p")  # overachievement profit
d1m = model_goal.addVar(lb=0, name="d1m")  # underachievement profit
d2p = model_goal.addVar(lb=0, name="d2p")  # overachievement emissions
d2m = model_goal.addVar(lb=0, name="d2m")  # underachievement emissions
d3p = model_goal.addVar(lb=0, name="d3p")  # overachievement labor
d3m = model_goal.addVar(lb=0, name="d3m")  # underachievement labor
T1, T2, T3 = 25000, 2000, 1500  # targets
model_goal.setObjective(d1m + d2p + d3m, GRB.MINIMIZE)
P = gp.quicksum(profit[i] * x_g[i] for i in range(n))
E = gp.quicksum(e[i] * x_g[i] for i in range(n))
L = gp.quicksum(l[i] * x_g[i] for i in range(n))
model_goal.addConstr(P + d1m - d1p == T1, "goal_profit")
model_goal.addConstr(E + d2m - d2p == T2, "goal_emissions")
model_goal.addConstr(L + d3m - d3p == T3, "goal_labor")
model_goal.addConstr(gp.quicksum(x_g[i] for i in range(n)) <= K, "capacity")
for i in range(n):
    model_goal.addConstr(x_g[i] <= d[i], f"demand_{i}")
model_goal.addConstr(gp.quicksum(c[i] * x_g[i] for i in range(n)) <= B, "budget")
model_goal.optimize()
if model_goal.status == GRB.OPTIMAL:
    print("\nGoal Programming Results:")
    print(f"Profit: {P.getValue():.2f}")
    print(f"Emissions: {E.getValue():.2f}")
    print(f"Labor: {L.getValue():.2f}")
    print("Deviations:")
    print(f"Profit underachievement: {d1m.X:.2f}")
    print(f"Emissions overachievement: {d2p.X:.2f}")
    print(f"Labor underachievement: {d3m.X:.2f}")
    for i in range(n):
        print(f"Product {i+1} production: {x_g[i].X:.2f} units")
else:
    print("Goal Programming: No optimal solution found.")