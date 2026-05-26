from gurobipy import *
import numpy as np
import pandas as pd
import time
 
def toType(var, vartype):
    
    if type(var) == np.ndarray:
        return var
    
    if vartype == "array":
        dims = tuple(max(idx) + 1 for idx in zip(*var.keys()))
        newvar = np.zeros(dims)
    else:        
        newvar = {}

    for it in var:
        if vartype == "decision variable":
            newvar[it] = var[it]
        elif vartype == "parameter":
            newvar[it] = var[it].X
        elif vartype == "array":
            newvar[it] = var[it].X
    return newvar

#@dataclass
class UCParams:
    def __init__(self, params):
        self.I = params["I"]
        self.T = params["T"]
        self.demand = params["demand"]
        self.id_list = params["id_list"]
        self.cost = params["cost"]
        self.fixed_cost = params["fixed_cost"]
        self.startup_cost = params["startup_cost"]
        self.production_max = params["production_max"]
        self.production_min = params["production_min"]

class Master():
    def __init__(self, params: UCParams):
        self.params = params
        I = self.params.I
        T = self.params.T

        fixed_cost = self.params.fixed_cost
        startup_cost = self.params.startup_cost

        self.model = Model("Master Problem")
        self.model.Params.OutputFlag = 0
        
        self.unit_on = self.model.addVars(I, T, vtype=GRB.BINARY, lb=0.0, ub=1.0, name="u")
        self.v_start = self.model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="v")
        self.phi = self.model.addVar(vtype=GRB.CONTINUOUS, name="phi")

        self.model.setObjective(
            quicksum(fixed_cost[i] * self.unit_on[i, t] + startup_cost[i] * self.v_start[i, t] for i in range(I) for t in range(T))
            + self.phi
            , GRB.MINIMIZE
        )

        self.model.addConstrs((self.unit_on[i, t] - self.unit_on[i, t-1] <= self.v_start[i, t]
                                for i in range(I) for t in range(1, T)), name="startup")

        self.model.update()

class Subproblem():
    # Dual variables for the subproblem. Is used both for optimality and feasibility
    def dual_vars(self):
        I = self.I
        T = self.T
        self.alpha = self.model.addVars(I, T, vtype=GRB.CONTINUOUS, name="alpha")
        self.beta = self.model.addVars(I, T, vtype=GRB.CONTINUOUS, name="beta")
        self.gamma = self.model.addVars(T, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="gamma")
        self.model.update()
    
    # Generates the expression for the subproblem objective function. It is used both for updating obejctive functions and generating cuts
    def sub_objective_expr(self, unit_on, usecase):
        # Gather parameters
        I = self.I
        T = self.T
        demand = self.params.demand
        production_max = self.params.production_max
        production_min = self.params.production_min

        dual_type = "decision variable" if usecase == "objective function" else "parameter"
        unit_type = "decision variable" if usecase == "benders cut" else "parameter"

        # Burde erstattes med en Base-function
        alpha = toType(self.alpha, dual_type)
        beta = toType(self.beta, dual_type)
        gamma = toType(self.gamma, dual_type)
        unit_on = toType(unit_on, unit_type)

        # Generate the expression
        expr = (- quicksum(alpha[i, t] * production_max[i] * unit_on[i, t] for i in range(I) for t in range(T))
                + quicksum( beta[i, t] * production_min[i] * unit_on[i, t] for i in range(I) for t in range(T))
                + quicksum(gamma[t] * demand[t] for t in range(T)))

        return expr

    # Constraints for the dual. Is used both for optimality and feasibility
    def dual_constraints(self, RHSasZero=False):
        I = self.I
        T = self.T
        cost = self.cost

        dual_constraints = self.model.addConstrs((-self.alpha[i, t] + self.beta[i, t] + self.gamma[t] <= cost[i]
                                             for i in range(I) for t in range(T)), name="dual")

        if RHSasZero:
            for i in range(I):
                for t in range(T):
                    dual_constraints[i, t].RHS = 0

        self.model.update()


class OptimalitySubproblem(Subproblem):
    def __init__(self, unit_on, params: UCParams):
        self.params = params
        self.I = self.params.I
        self.T = self.params.T
        self.cost = self.params.cost

        self.model = Model("Optimality Problem")
        self.model.Params.OutputFlag = 0

        self.dual_vars()

        obj_expr = self.sub_objective_expr(unit_on, "objective function")
        self.model.setObjective(obj_expr, GRB.MAXIMIZE)

        self.dual_constraints()
        self.model.update()

class FeasibilitySubproblem(Subproblem):
    def __init__(self, unit_on, params: UCParams):
        self.params = params
        self.I = self.params.I
        self.T = self.params.T
        self.cost = self.params.cost

        I = self.I
        T = self.T

        self.model = Model("Feasibility Problem")
        self.model.Params.OutputFlag = 0

        self.dual_vars()

        self.gamma_plus = self.model.addVars(T, vtype=GRB.CONTINUOUS, name="gamma^+")
        self.gamma_minus = self.model.addVars(T, vtype=GRB.CONTINUOUS, name="gamma^-")

        obj_expr = self.sub_objective_expr(unit_on, "objective function")
        self.model.setObjective(obj_expr, GRB.MAXIMIZE)

        self.dual_constraints(RHSasZero=True)

        self.model.addConstr(quicksum(self.alpha[i, t] for i in range(I) for t in range(T)) +
                             quicksum(self.beta[i, t] for i in range(I) for t in range(T)) +
                             quicksum(self.gamma_plus[t] + self.gamma_minus[t] for t in range(T))
                             == 1, name="normalize")

        self.model.addConstrs((self.gamma[t] == self.gamma_plus[t] - self.gamma_minus[t] for t in range(T)), name="gammaconstr")

        self.model.update()

class Benders:
    def __init__(self, params: UCParams):
        self.params = params

        self.I = self.params.I
        self.T = self.params.T
        
        self.master = Master(params)

        self.phi = self.master.phi
        self.unit_on = self.master.unit_on

        zero_unit_on = np.zeros((self.I, self.T))

        self.opti = OptimalitySubproblem(zero_unit_on, params)
        self.feas = FeasibilitySubproblem(zero_unit_on, params)

        self.tolerance = 1e-8

        self.feas_tol = self.tolerance
        self.opti_tol = self.tolerance


        self.violation = 0

        self.timeout = None

        self.inBnB = False

    def benders_cut(self, unit_data, phi):
        benders_cut_start_time = time.time()

        # Feasibility test
        feas_objective_expr = self.feas.sub_objective_expr(unit_data, "objective function")
        self.feas.model.setObjective(feas_objective_expr, GRB.MAXIMIZE)
        self.feas.model.update()

        self.feas.model.optimize()

        # If F(u^k) < 0 then the solution is infeasible
        if self.feas.model.ObjVal > 0 + self.feas_tol:

            # Add feasibility cut
            cut_expr = self.feas.sub_objective_expr(self.unit_on, "benders cut") <= 0

            # Save data
            self.rows.append({
                "cut_type": "feasibility cut",
                "cut_start_time": benders_cut_start_time,
                "cur_time": time.time(),
                "opti_runtime": self.opti.model.Runtime,
                "feas_runtime": self.feas.model.Runtime,
                "violation": self.feas.model.ObjVal,
                "master_nodes": self.nodecount
            })
            print("Adding feasibility cut")

            return "feasibility cut", cut_expr
        
        # If unit_on is continous and the solution is feasible then unit_on should be made binary
        if self.master.unit_on[0,0].vtype == GRB.CONTINUOUS:
            return "Master to MIP", None

        # Optimality test
        opti_objective_expr = self.opti.sub_objective_expr(unit_data, "objective function")
        self.opti.model.setObjective(opti_objective_expr, GRB.MAXIMIZE)
        self.opti.model.update()
        self.opti.model.optimize()

        # If Q^D(u^k) > \phi then the solution is not optimal
        if self.opti.model.ObjVal > phi + self.opti_tol:

            # Add optimality cut
            cut_expr = self.opti.sub_objective_expr(self.unit_on, "benders cut") <= self.phi

            # Save data
            self.rows.append({
                "cut_type": "optimality cut",
                "cut_start_time": benders_cut_start_time,
                "cur_time": time.time(),
                "opti_runtime": self.opti.model.Runtime,
                "feas_runtime": self.feas.model.Runtime,
                "violation": self.opti.model.ObjVal - phi,
                "master_nodes": self.nodecount
            })
            print("Adding optimality cut")
            return "optimality cut", cut_expr

        # If F(u^k) >= 0 and Q^D(u^k) <= \phi then the solution is optimal
        # Save data
        self.rows.append({
            "cut_type": "No cut",
            "cut_start_time": benders_cut_start_time,
            "cur_time": time.time(),
            "opti_runtime": self.opti.model.Runtime,
            "feas_runtime": self.feas.model.Runtime,
            "violation": self.opti.model.ObjVal - phi,
            "master_nodes": self.nodecount
        })
        print("No cut added. Solution is optimal")
        return "Converged", None
        

    def benders_algorithm(self, max_iterations, Master_LP_start=True):
        status = GRB.ITERATION_LIMIT
        self.rows = []

        # Sets unit_on to be contoues for the generation of feasibility cuts
        if Master_LP_start:
            for it in self.master.unit_on:
                self.master.unit_on[it].vtype = GRB.CONTINUOUS
            self.master.model.update()

        # Time at start of algorithm
        time_0 = time.time()

        # Initial solution of Master Problem
        self.master.model.optimize()

        k = 0
        while k < max_iterations:

            self.nodecount = self.master.model.NodeCount

            if self.timeout is not None:
                if time.time() - time_0 > self.timeout:
                    print("Timeout reached. Stopping algorithm.")
                    status = GRB.TIME_LIMIT
                    break

            # Generating af dataframe of unit_on-values
            unit_data = np.zeros((self.I,self.T))
            for it in self.unit_on:
                unit_data[it] = self.unit_on[it].X

            # Generates benders cut
            cut_type, cut_expr = self.benders_cut(unit_data, self.phi.X)

            # If unit_on is continous and the solution is feasible then unit_on should be made binary
            if cut_type == "Master to MIP":
                for it in self.master.unit_on:
                    self.master.unit_on[it].vtype = GRB.BINARY
                self.master.model.update()

                print("Master is now MIP")

                self.master.model.optimize()
                continue

            # If the problem has converged, the algorithm stops
            if cut_type == "Converged":
                print("Problem converged.")
                status = GRB.OPTIMAL
                break

            # Adds the constraint to the Master Problem
            self.master.model.addConstr(cut_expr, name=f"{cut_type}_{k}")
            self.master.model.update()

            k += 1

            self.master.model.optimize()

        run_data = pd.DataFrame(self.rows)
        run_data["time"] = run_data["cur_time"] - time_0

        return status, run_data


    # Callback function for doing benders cuts in branch and bound
    def benders_callback(self, model, where):
        
         # We only want to do callback when an integer solution is found
        if where != GRB.Callback.MIPSOL:
            return
        
        self.nodecount = model.cbGet(GRB.Callback.MIPSOL_NODCNT)

        # Get the current value of the unit_on variable
        unit_data = np.zeros((self.I,self.T))
        for it in self.unit_on:
            unit_data[it] = model.cbGetSolution(self.unit_on[it])

        # Get the current value of \phi
        phi = model.cbGetSolution(self.phi)

        cut_type, cut_expr = self.benders_cut(unit_data, phi)

        if cut_type == "Converged":
            return

        model.cbLazy(cut_expr)
        return

    def Solve_BaB(self):
        self.master.model.Params.LazyConstraints = 1
        self.master.model.Params.InfUnbdInfo = 1
        self.master.model.Params.DualReductions = 0
        self.master.model.Params.OutputFlag = 1

        self.inBnB = True
        self.feas_tol = 0

        self.rows = []

        if self.timeout is not None:
            self.master.model.Params.TimeLimit = self.timeout

        time_0 = time.time()
        
        self.master.model.optimize(self.benders_callback)

        run_data = pd.DataFrame(self.rows)

        run_data["time"] = run_data["cur_time"] - time_0

        return self.master.model.Status, run_data