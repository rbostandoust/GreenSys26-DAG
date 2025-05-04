import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon
import json
from ortools.sat.python import cp_model
import math

def generate_multiple_job_schedules(job_dict, sample_num, num_jobs, num_machines, seed=42):
    """
    Run `sample_num` repeated samplings of `num_jobs` from job_dict.
    Each sample produces a list of job schedules (nested machine-duration tuples).
    All outputs are deterministic due to seeding.
    
    Returns:
        - job_schedules: List of length sample_num. Each element is a job schedule.
        - all_selected_job_ids: List of length sample_num. Each element is the list of job IDs for that sample.
    """
    all_job_ids = list(job_dict.keys())
    if num_jobs > len(all_job_ids):
        raise ValueError("num_jobs cannot be larger than the number of available jobs in job_dict")

    job_schedules = []
    all_selected_job_ids = []

    # Use a reproducible seeded random generator
    rng = random.Random(seed)

    for i in range(sample_num):
        # Derive a unique seed per sample to get deterministic but varied results
        sample_rng = random.Random(seed + i)
        selected_job_ids = sample_rng.sample(all_job_ids, num_jobs)

        schedule = []
        for job_id in selected_job_ids:
            durations = job_dict[job_id]["operations_duration"]
            operations = [
                [(machine_id, duration) for machine_id in range(num_machines)]
                for duration in durations
            ]
            schedule.append(operations)

        job_schedules.append(schedule)
        all_selected_job_ids.append(selected_job_ids)

    return job_schedules, all_selected_job_ids

def makespan_minimizer(jobs_data, num_machines, jobs_arrival_epoch, initial_horizon):
    print(f"Arrival epochs = {jobs_arrival_epoch}")
    model = cp_model.CpModel()

    all_tasks = {}
    all_machines = [[] for _ in range(num_machines)]
    # Create variables
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            for m_id, duration in alternatives:
                suffix = f'_{job_id}_{task_id}_{m_id}'
                start = model.NewIntVar(0, initial_horizon, 'start' + suffix)
                end = model.NewIntVar(0, initial_horizon, 'end' + suffix)
                presence = model.NewBoolVar('presence' + suffix)
                interval = model.NewOptionalIntervalVar(start, duration, end, presence, 'interval' + suffix)

                all_tasks[(job_id, task_id, m_id)] = (start, end, interval, presence)
                all_machines[m_id].append((start, duration, interval))

    # Add constraints
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            # Only one machine must be selected for each operation
            presences = [all_tasks[(job_id, task_id, m_id)][3] for m_id, _ in alternatives]
            model.AddExactlyOne(presences)

            # Precedence constraints between operations
            if task_id > 0:
                prev_alts = jobs_data[job_id][task_id - 1]
                curr_alts = alternatives
                for prev_m_id, _ in prev_alts:
                    for curr_m_id, _ in curr_alts:
                        prev_end = all_tasks[(job_id, task_id - 1, prev_m_id)][1]
                        curr_start = all_tasks[(job_id, task_id, curr_m_id)][0]
                        prev_presence = all_tasks[(job_id, task_id - 1, prev_m_id)][3]
                        curr_presence = all_tasks[(job_id, task_id, curr_m_id)][3]
                        model.Add(curr_start >= prev_end).OnlyEnforceIf([prev_presence, curr_presence])
            # first task's start time must be after arrival time
            if task_id == 0:
                curr_alts = alternatives
                for curr_m_id, _ in curr_alts:
                    curr_presence = all_tasks[(job_id, task_id, curr_m_id)][3]
                    curr_start = all_tasks[(job_id, task_id, curr_m_id)][0]
                    model.Add(curr_start >= jobs_arrival_epoch[job_id]).OnlyEnforceIf(curr_presence)
            
            

    # Machine capacity constraints: no overlapping intervals
    for machine_id in range(num_machines):
        machine_tasks = all_machines[machine_id]
        model.AddNoOverlap([interval for _, _, interval in machine_tasks]) # For every pair of intervals in the list, the solver ensures that if both are present, then their execution windows do not overlap

    # Objective: minimize makespan (latest end time)
    all_ends = [end for (_, end, _, _) in all_tasks.values()]
    makespan = model.NewIntVar(0, initial_horizon, 'makespan')
    model.AddMaxEquality(makespan, all_ends)
    model.Minimize(makespan)

    # Solve model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    solver.parameters.max_time_in_seconds = initial_horizon
    status = solver.Solve(model)

    # Output solution
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        total_carbon_consumption = 0
        print(f'Minimized makespan: {solver.Value(makespan)}')
        for (j, t, m), (start, end, interval, presence) in all_tasks.items():
            if solver.Value(presence):
                print(f'Job {j}, Task {t} on Machine {m} → Start: {solver.Value(start)}, End: {solver.Value(end)}')
                total_carbon_consumption += sum(carbon_trace_spec_day_per_epoch[location][solver.Value(start):solver.Value(end)]) * (epoch_in_minutes / 60)
        print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g")
        return solver.Value(makespan), "Success"
    elif status == cp_model.INFEASIBLE:
        print("❌ Problem is infeasible.")
        return None, "Infeasible"
    else:
        print("❓ Solver stopped (possibly due to timeout) with status:", status)
        return None, "TimerInterrupt"

def carbon_aware_scheduling(jobs_data, max_allowed_makespan, minimum_makespan, makespan_activated, solver_max_timeout_in_seconds):
    model = cp_model.CpModel()
    if makespan_activated:
        horizon = min(max_allowed_makespan, initial_horizon)
        if horizon == max_allowed_makespan:
            print(f"max allowed makespan = {horizon} -> {max_allowed_makespan / minimum_makespan} x minimum makespan")
        else:
            print(f"max allowed makespan = {horizon / (24 * (60 // epoch_in_minutes))} day")
    else:
        horizon = initial_horizon
        print(f"max allowed makespan = {horizon / (24 * (60 // epoch_in_minutes))} day")
    print(f"Arrival epochs = {jobs_arrival_epoch}")
    all_tasks = {}
    all_machines = [[] for _ in range(num_machines)]

    # Create variables
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            for m_id, duration in alternatives:
                suffix = f'_{job_id}_{task_id}_{m_id}'
                start = model.NewIntVar(0, horizon, 'start' + suffix)
                end = model.NewIntVar(0, horizon, 'end' + suffix)
                presence = model.NewBoolVar('presence' + suffix)
                interval = model.NewOptionalIntervalVar(start, duration, end, presence, 'interval' + suffix)

                all_tasks[(job_id, task_id, m_id)] = (start, end, interval, presence)
                all_machines[m_id].append((start, duration, interval, presence))

    # Add constraints
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            # Only one machine must be selected for each operation
            presences = [all_tasks[(job_id, task_id, m_id)][3] for m_id, _ in alternatives]
            model.AddExactlyOne(presences)

            # Precedence constraints between operations
            if task_id > 0:
                prev_alts = jobs_data[job_id][task_id - 1]
                curr_alts = alternatives
                for prev_m_id, _ in prev_alts:
                    for curr_m_id, _ in curr_alts:
                        prev_end = all_tasks[(job_id, task_id - 1, prev_m_id)][1]
                        curr_start = all_tasks[(job_id, task_id, curr_m_id)][0]
                        prev_presence = all_tasks[(job_id, task_id - 1, prev_m_id)][3]
                        curr_presence = all_tasks[(job_id, task_id, curr_m_id)][3]
                        model.Add(curr_start >= prev_end).OnlyEnforceIf([prev_presence, curr_presence])
            # first task's start time must be after arrival epoch
            if task_id == 0:
                curr_alts = alternatives
                for curr_m_id, _ in curr_alts:
                    curr_presence = all_tasks[(job_id, task_id, curr_m_id)][3]
                    curr_start = all_tasks[(job_id, task_id, curr_m_id)][0]
                    model.Add(curr_start >= jobs_arrival_epoch[job_id]).OnlyEnforceIf(curr_presence)
            
            

    # Machine capacity constraints: no overlapping intervals
    for machine_id in range(num_machines):
        machine_tasks = all_machines[machine_id]
        model.AddNoOverlap([interval for _, _, interval, _ in machine_tasks]) # For every pair of intervals in the list, the solver ensures that if both are present, then their execution windows do not overlap

    # Limiting makespan by an upper-bound
    if makespan_activated:
        all_ends = [end for (_, end, _, _) in all_tasks.values()]
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, all_ends)
        model.Add(makespan <= max_allowed_makespan)
    
    
    # Carbon-aware objective
    total_carbon_terms = []
    active_vars = []
    b_vars = []
    for t in range(horizon):
        active_vars.append([])
        for m in range(num_machines):
            active = model.NewBoolVar(f'active_{t}_{m}')
            active_vars[t].append(active)
            relevant_presences = []
            for (start, dur, interval, presence) in all_machines[m]:
                b = model.NewBoolVar(f'running_{t}_{m}') # b[t][m] = True when a task is running on machine m and the time t is within the [start, end] of the task                
                in_window1 = model.NewBoolVar('')
                in_window2 = model.NewBoolVar('')
                model.Add(start <= t).OnlyEnforceIf(in_window1)
                model.Add(start > t).OnlyEnforceIf(in_window1.Not())
                model.Add(t < start + dur).OnlyEnforceIf(in_window2)
                model.Add(t >= start + dur).OnlyEnforceIf(in_window2.Not())
                model.AddImplication(b, in_window1)
                model.AddImplication(b, in_window2)
                model.AddImplication(b, presence)
                model.AddBoolOr(in_window1.Not(), in_window2.Not(), presence.Not(), b)
                relevant_presences.append(b)
            b_vars.append(relevant_presences)
            
            model.AddMaxEquality(active, relevant_presences) # if one of the b s is 1 then active[t][m] = 1

            # Compute carbon emission at this time step
            carbon_value = carbon_trace_spec_day_per_epoch[location][t] * power
            carbon_term = model.NewIntVar(0, carbon_value, f'carbon_{t}_{m}')
            model.Add(carbon_term == carbon_value).OnlyEnforceIf(active)
            model.Add(carbon_term == 0).OnlyEnforceIf(active.Not())
            total_carbon_terms.append(carbon_term)

    total_carbon = model.NewIntVar(0, 1000000, 'total_carbon')
    model.Add(total_carbon == sum(total_carbon_terms))
    model.Minimize(total_carbon)

    # Solve model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_max_timeout_in_seconds
    status = solver.Solve(model)

    # Output solution
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        total_carbon_consumption = solver.Value(total_carbon) * (epoch_in_minutes / 60)
        if makespan_activated:
            print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g with maximum allowed makespan = {max_allowed_makespan}")
        else:
            print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g with 1 days limit")
        if status == cp_model.OPTIMAL:
            print(f"✅ Solution is also optimal!")
        for (j, t, m), (start, end, _, presence) in all_tasks.items():
            if solver.Value(presence):
                print(f'Job {j}, Task {t} on Machine {m} → Start: {solver.Value(start)}, End: {solver.Value(end)}')
        return total_carbon_consumption, "Success"
    elif status == cp_model.INFEASIBLE:
        print("❌ Problem is infeasible.")
        return None, "Infeasible"
    else:
        print("❓ Solver stopped (possibly due to timeout) with status:", status)
        return None, "TimerInterrupt"

"""
General vars
"""
epoch_in_minutes = 5
initial_horizon = 24 * 60 // epoch_in_minutes # 1 day - This is the maximum slack
solver_max_timeout_in_seconds = 1 * 60

"""
Carbon Intensity
"""
selected_date = pd.to_datetime("2024-06-01").date()
location = "California"

carbon_trace = {}
carbon_trace_spec_day = {}
carbon_trace_spec_day_per_epoch = {}
df1 = pd.read_csv(f"../CarbonTrace/US-CAL-CISO.csv")[['datetime', 'carbon_intensity_avg']] # 2020-2023
df2 = pd.read_csv(f"../CarbonTrace/US-CAL-CISO-2024.csv")[['Datetime (UTC)', 'Carbon intensity gCO₂eq/kWh (Life cycle)']] # 2024
df2.rename(columns={"Datetime (UTC)": "datetime", "Carbon intensity gCO₂eq/kWh (Life cycle)": "carbon_intensity_avg"}, inplace=True)
carbon_trace[location] = pd.concat([df1, df2], ignore_index=True)
carbon_trace[location]['datetime'] = carbon_trace[location]['datetime'].apply(
    lambda x: pd.to_datetime(x, utc=True, errors='coerce')
)
next_date = selected_date + pd.Timedelta(days=1)
date_windows = [selected_date]
for extra_day in range(7): # one week of intensity starting from selected date 
    next_date = selected_date + pd.Timedelta(days=extra_day)
    date_windows.append(next_date)
mask = carbon_trace[location]["datetime"].dt.date.isin(date_windows)
carbon_trace_spec_day[location] = carbon_trace[location][mask].copy()
carbon_trace_spec_day[location]["Hour"] = carbon_trace[location]["datetime"].dt.hour

carbon_trace_spec_day_per_epoch[location] = []
epoch_num_in_one_hour = 60 // epoch_in_minutes
intensities = carbon_trace_spec_day[location]['carbon_intensity_avg'].tolist()
for intensity in intensities:
    for i in range(epoch_num_in_one_hour):
        carbon_trace_spec_day_per_epoch[location].append(int(round(intensity)))
        
"""
Sampling from the job pool and determining arrival epochs
"""
num_instances = 1000
num_jobs = 10 # per instance
num_operations_per_job = 3
mean_duration_per_op_in_epoch = 3
num_machines = 5 # per instance
power = 1 # machines are homogeneous
jobs_arrival_epoch = [0 for _ in range(num_jobs)] # ever job arrives hour0 of the day
with open(f"../Data/JobPool/JobPool_{num_operations_per_job}Ops_MeanOpDur={mean_duration_per_op_in_epoch}_Epoch={epoch_in_minutes}.json", "r") as f:
    job_dict = json.load(f)
list_jobs_data, list_job_ids = generate_multiple_job_schedules(job_dict, num_jobs = num_jobs, num_machines = num_machines, sample_num = num_instances, seed = 42)
print(list_jobs_data[0])
print(list_job_ids[0])

"""
Running the flexible jobshop scenario for each instance
"""
instance_num_start, instance_num_end = 0, num_instances
candidate_makespan_slack_coeff = [1, 2, 5, 10]
for instance_num in range(instance_num_start, instance_num_end, 1):
    # run makespan minimizer
    print(f"Running Instance {instance_num}")
    global_minimum_makespan, solver_status = makespan_minimizer(jobs_data = list_jobs_data[instance_num], 
                                                                num_machines = num_machines,
                                                                jobs_arrival_epoch = jobs_arrival_epoch, 
                                                                initial_horizon = initial_horizon)
    if solver_status != "Success":
        continue
    # carbon-aware scheduling
    slack_coeff = 10
    max_allowed_makespan = int(math.ceil(slack_coeff * global_minimum_makespan)) # in epoch
    makespan_activated = False
    carbon_consumption, solver_status = carbon_aware_scheduling(jobs_data = list_jobs_data[instance_num], 
                                                                max_allowed_makespan = max_allowed_makespan, 
                                                                minimum_makespan = global_minimum_makespan, 
                                                                makespan_activated = makespan_activated, 
                                                                solver_max_timeout_in_seconds = solver_max_timeout_in_seconds)
    break
    
