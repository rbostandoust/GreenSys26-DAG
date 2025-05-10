import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon
import json
from ortools.sat.python import cp_model
import math
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

def generate_multiple_job_arrival_epochs(start_hour, end_hour, epoch_in_minutes, sample_num, num_jobs, seed):
    local_rng = random.Random(seed)
    list_jobs_arrival_epoch = []
    for _ in range(sample_num):
        jobs_arrival_epoch = sorted([local_rng.randrange(start_hour * 60 // epoch_in_minutes, end_hour * 60 // epoch_in_minutes) for _ in range(num_jobs)])
        list_jobs_arrival_epoch.append(jobs_arrival_epoch)
    return list_jobs_arrival_epoch
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

def append_to_csv(data_list, file_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    new_df = pd.DataFrame(data_list)

    if os.path.exists(file_path):
        # Load existing CSV and append new data
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # File doesn't exist, just use the new DataFrame
        combined_df = new_df

    # Save back to CSV
    combined_df.to_csv(file_path, index=False)

def generate_instance_carbon_intensity_trace(selected_date, location, windows_size_in_days = 30):
    carbon_trace_spec_day = {}
    carbon_trace_spec_day_per_epoch = {}
    date_windows = [selected_date]
    for extra_day in range(windows_size_in_days): # max of 1 month of intensity starting from selected date 
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
    return carbon_trace_spec_day_per_epoch
    
def makespan_minimizer(carbon_trace_spec_day_per_epoch, jobs_data, jobs_id, num_machines, jobs_arrival_epoch, initial_horizon, solver_max_timeout_in_seconds):
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
                adjusted_duration = int(math.ceil(duration_coeff[m_id] * duration))
                interval = model.NewOptionalIntervalVar(start, adjusted_duration, end, presence, 'interval' + suffix)

                all_tasks[(job_id, task_id, m_id)] = (start, end, interval, presence)
                all_machines[m_id].append((start, adjusted_duration, interval))

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
    solver.parameters.max_time_in_seconds = solver_max_timeout_in_seconds
    status = solver.Solve(model)

    # Output solution
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        total_carbon_consumption, total_energy_consumption = 0, 0
        machines_status = {}
        # print(f'Minimized makespan: {solver.Value(makespan)}')
        for (j, t, m), (start, end, interval, presence) in all_tasks.items():
            if solver.Value(presence):
                # print(f'Job {j}, Task {t} on Machine {m} → Start: {solver.Value(start)}, End: {solver.Value(end)}')
                total_carbon_consumption += sum(carbon_trace_spec_day_per_epoch[solver.Value(start):solver.Value(end)]) * power[m] * (epoch_in_minutes / 60)
                total_energy_consumption += (solver.Value(end) - solver.Value(start)) * power[m] * (epoch_in_minutes / 60)
                if f"Server{m}" not in machines_status.keys():
                    machines_status[f"Server{m}"] = []
                machines_status[f"Server{m}"].append([int(jobs_id[j]), t, solver.Value(start), solver.Value(end)]) # [job, task, start, end]
        print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g, total energy consumed = {total_energy_consumption:.2f}kWh")
        machines_status = dict(sorted(machines_status.items(), key=lambda item: int(item[0].replace('Server', ''))))
        return solver.Value(makespan), total_carbon_consumption, total_energy_consumption, "Success", machines_status
    elif status == cp_model.INFEASIBLE:
        print("❌ Problem is infeasible.")
        return None, None, None, "Infeasible", _
    else:
        print("❓ Solver stopped (possibly due to timeout) with status:", status)
        return None, None, None, "TimerInterrupt", _

def energy_aware_scheduling(carbon_trace_spec_day_per_epoch, jobs_data,jobs_id, jobs_arrival_epoch, max_allowed_makespan, solver_max_timeout_in_seconds):
    model = cp_model.CpModel()
    all_tasks = {}
    all_machines = [[] for _ in range(num_machines)]

    # Create variables
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            for m_id, duration in alternatives:
                suffix = f'_{job_id}_{task_id}_{m_id}'
                start = model.NewIntVar(0, max_allowed_makespan, 'start' + suffix)
                end = model.NewIntVar(0, max_allowed_makespan, 'end' + suffix)
                presence = model.NewBoolVar('presence' + suffix)
                adjusted_duration = int(math.ceil(duration_coeff[m_id] * duration))
                interval = model.NewOptionalIntervalVar(start, adjusted_duration, end, presence, 'interval' + suffix)

                all_tasks[(job_id, task_id, m_id)] = (start, end, interval, presence)
                all_machines[m_id].append((start, adjusted_duration, interval, presence))

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
    all_ends = [end for (_, end, _, _) in all_tasks.values()]
    makespan = model.NewIntVar(0, max_allowed_makespan, 'makespan')
    model.AddMaxEquality(makespan, all_ends)
    model.Add(makespan <= max_allowed_makespan)
    
    
    # energy-aware objective
    # total_carbon_terms = []
    total_energy_terms = []
    active_vars = []
    b_vars = []
    for t in range(max_allowed_makespan):
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
            # carbon_value = carbon_trace_spec_day_per_epoch[t] * power[m]
            if not mixed_objective:
                energy_value =  power[m]
            else:
                alpha = 0.01
                energy_value = (20 + int(round(carbon_trace_spec_day_per_epoch[t] * alpha))) * power[m]
            # carbon_term = model.NewIntVar(0, carbon_value, f'carbon_{t}_{m}')
            energy_term = model.NewIntVar(0, energy_value, f'energy_{t}_{m}')
            # model.Add(carbon_term == carbon_value).OnlyEnforceIf(active)
            model.Add(energy_term == energy_value).OnlyEnforceIf(active)
            model.Add(energy_term == 0).OnlyEnforceIf(active.Not())
            total_energy_terms.append(energy_term)

    # total_carbon = model.NewIntVar(0, 1000000, 'total_carbon')
    total_energy = model.NewIntVar(0, 1000000, 'total_energy')
    model.Add(total_energy == sum(total_energy_terms))
    model.Minimize(total_energy)

    # Solve model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_max_timeout_in_seconds
    status = solver.Solve(model)

    # Output solution
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # total_energy_consumption = solver.Value(total_energy) * (epoch_in_minutes / 60)
        total_energy_consumption = 0
        total_carbon_consumption = 0        
        machines_status = {}
        makespan_val = 0
        if status == cp_model.OPTIMAL:
            print(f"✅ Solution is also optimal!")
        for (j, t, m), (start, end, _, presence) in all_tasks.items():
            if solver.Value(presence):
                total_carbon_consumption += sum(carbon_trace_spec_day_per_epoch[solver.Value(start):solver.Value(end)]) * power[m] * (epoch_in_minutes / 60)
                total_energy_consumption += (solver.Value(end)-solver.Value(start)) * power[m] * (epoch_in_minutes / 60)
                # print(f'Job {j}, Task {t} on Machine {m} → Start: {solver.Value(start)}, End: {solver.Value(end)}')
                if f"Server{m}" not in machines_status.keys():
                    machines_status[f"Server{m}"] = []
                makespan_val = max(makespan_val, solver.Value(end))
                machines_status[f"Server{m}"].append([int(jobs_id[j]), t, solver.Value(start), solver.Value(end)]) # [job, task, start, end]
        machines_status = dict(sorted(machines_status.items(), key=lambda item: int(item[0].replace('Server', ''))))
        print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g, total energy consumption: {total_energy_consumption:.2f}kWh")
        return total_carbon_consumption, total_energy_consumption, makespan_val, "Success", machines_status
    elif status == cp_model.INFEASIBLE:
        print("❌ Problem is infeasible.")
        return None, None, None, "Infeasible", _
    else:
        print("❓ Solver stopped (possibly due to timeout) with status:", status)
        return None, None, None, "TimerInterrupt", _

def carbon_aware_scheduling(carbon_trace_spec_day_per_epoch, jobs_data,jobs_id, jobs_arrival_epoch, max_allowed_makespan, solver_max_timeout_in_seconds):
    model = cp_model.CpModel()
    all_tasks = {}
    all_machines = [[] for _ in range(num_machines)]

    # Create variables
    for job_id, job in enumerate(jobs_data):
        for task_id, alternatives in enumerate(job):
            for m_id, duration in alternatives:
                suffix = f'_{job_id}_{task_id}_{m_id}'
                start = model.NewIntVar(0, max_allowed_makespan, 'start' + suffix)
                end = model.NewIntVar(0, max_allowed_makespan, 'end' + suffix)
                presence = model.NewBoolVar('presence' + suffix)
                adjusted_duration = int(math.ceil(duration_coeff[m_id] * duration))
                interval = model.NewOptionalIntervalVar(start, adjusted_duration, end, presence, 'interval' + suffix)

                all_tasks[(job_id, task_id, m_id)] = (start, end, interval, presence)
                all_machines[m_id].append((start, adjusted_duration, interval, presence))

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
    all_ends = [end for (_, end, _, _) in all_tasks.values()]
    makespan = model.NewIntVar(0, max_allowed_makespan, 'makespan')
    model.AddMaxEquality(makespan, all_ends)
    model.Add(makespan <= max_allowed_makespan)
    
    
    # carbon-aware objective
    total_carbon_terms = []
    active_vars = []
    b_vars = []
    for t in range(max_allowed_makespan):
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
            if not mixed_objective:
                carbon_value = carbon_trace_spec_day_per_epoch[t] * power[m]
            else:
                alpha = 1
                carbon_value = (carbon_trace_spec_day_per_epoch[t] + alpha) * power[m]
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
        # total_carbon_consumption = solver.Value(total_carbon) * (epoch_in_minutes / 60)
        total_carbon_consumption = 0
        total_energy_consumption = 0
        machines_status = {}
        makespan_val = 0
        if status == cp_model.OPTIMAL:
            print(f"✅ Solution is also optimal!")
        for (j, t, m), (start, end, _, presence) in all_tasks.items():
            if solver.Value(presence):
                # print(f'Job {j}, Task {t} on Machine {m} → Start: {solver.Value(start)}, End: {solver.Value(end)}')
                if f"Server{m}" not in machines_status.keys():
                    machines_status[f"Server{m}"] = []
                total_carbon_consumption += sum(carbon_trace_spec_day_per_epoch[solver.Value(start):solver.Value(end)]) * power[m] * (epoch_in_minutes / 60)
                total_energy_consumption += (solver.Value(end)-solver.Value(start)) * power[m] * (epoch_in_minutes / 60)
                makespan_val = max(makespan_val, solver.Value(end))
                machines_status[f"Server{m}"].append([int(jobs_id[j]), t, solver.Value(start), solver.Value(end)]) # [job, task, start, end]
        machines_status = dict(sorted(machines_status.items(), key=lambda item: int(item[0].replace('Server', ''))))
        print(f"✅ Total carbon emitted: {total_carbon_consumption:.2f} g, total energy consumption: {total_energy_consumption:.2f}kWh")
        return total_carbon_consumption, total_energy_consumption, makespan_val, "Success", machines_status
    elif status == cp_model.INFEASIBLE:
        print("❌ Problem is infeasible.")
        return None, None, None, "Infeasible", _
    else:
        print("❓ Solver stopped (possibly due to timeout) with status:", status)
        return None, None, None, "TimerInterrupt", _

def run_objective_aware_task(carbon_trace_spec_day_per_epoch, selected_date, instance_num, max_allowed_makespan, slack_coeff, minimum_makespan, start_time,
                          list_jobs_data, list_job_ids, list_jobs_arrival_epoch,
                          num_jobs, num_machines, num_operations_per_job, power,
                          epoch_in_minutes, solver_max_timeout_in_seconds, log_dict_list_shared, experiment_type):
    if experiment_type == "Heterogen_Energy" or experiment_type == "Homogen_Energy":
        carbon_consumption, energy_consumption, makespan, solver_status, servers_status = energy_aware_scheduling(
            carbon_trace_spec_day_per_epoch = carbon_trace_spec_day_per_epoch,
            jobs_data=list_jobs_data[instance_num],
            jobs_id=list_job_ids[instance_num],
            jobs_arrival_epoch = list_jobs_arrival_epoch[instance_num],
            max_allowed_makespan=max_allowed_makespan,
            solver_max_timeout_in_seconds=solver_max_timeout_in_seconds
        )
    else:
        carbon_consumption, energy_consumption, makespan, solver_status, servers_status = carbon_aware_scheduling(
            carbon_trace_spec_day_per_epoch = carbon_trace_spec_day_per_epoch,
            jobs_data=list_jobs_data[instance_num],
            jobs_id=list_job_ids[instance_num],
            jobs_arrival_epoch = list_jobs_arrival_epoch[instance_num],
            max_allowed_makespan=max_allowed_makespan,
            solver_max_timeout_in_seconds=solver_max_timeout_in_seconds
        )

    if solver_status == "Success":
        log = {
            'ElapsedTime': str(datetime.now() - start_time).split(".")[0], "Datetime": selected_date,
            'Instance': instance_num, 'IsCarbonAware': True,
            'MaxMakeSpan': max_allowed_makespan, 'MinMakeSpan': minimum_makespan, 'SlackCoeff': slack_coeff, 'Makespan': makespan,
            'CarbonConsumption(g)': round(carbon_consumption, 2), 'EnergyConsumption(kWh)': round(energy_consumption, 2),
            'JobNumber': num_jobs, 'ServerNumber': num_machines, 'OperationsPerJob': num_operations_per_job,
            'ServerPower': power, 'EpochDuration(min)': epoch_in_minutes, 'SolverTimer(min)': solver_max_timeout_in_seconds // 60,
            'JobIndex': list_job_ids[instance_num], 'JobArrivalEpoch': list_jobs_arrival_epoch[instance_num]
        }
        for sid in range(num_machines):
            log[f"Server{sid}"] = servers_status.get(f"Server{sid}", [])
        log_dict_list_shared.append(log)
"""
General vars
"""
epoch_in_minutes = 15
initial_horizon = 7 * 24 * 60 // epoch_in_minutes # 1 week horizon for makespan minimizer
solver_max_timeout_in_seconds = 1 * 60

"""
Carbon Intensity
"""
# location = "California"
location = "AU-SA"

# loading the whole th trace
carbon_trace = {}
if location == "California":
    df2 = pd.read_csv(f"../CarbonTrace/US-CAL-CISO-2024.csv")[['Datetime (UTC)', 'Carbon intensity gCO₂eq/kWh (Life cycle)']] # 2024
elif location == "AU-SA":
    df2 = pd.read_csv(f"../CarbonTrace/AU-SA_2024.csv")[['Datetime (UTC)', 'Carbon intensity gCO₂eq/kWh (Life cycle)']] # 2024
else:
    raise Exception ("Unknown Location!")
# df2 = pd.read_csv(f"../CarbonTrace/AU-WA-2024.csv")[['Datetime (UTC)', 'Carbon intensity gCO₂eq/kWh (Life cycle)']] # 2024
df2.rename(columns={"Datetime (UTC)": "datetime", "Carbon intensity gCO₂eq/kWh (Life cycle)": "carbon_intensity_avg"}, inplace=True)
carbon_trace[location] = df2.copy()
carbon_trace[location]['datetime'] = carbon_trace[location]['datetime'].apply(
    lambda x: pd.to_datetime(x, utc=True, errors='coerce')
)
        
"""
Sampling from the job pool and determining arrival epochs
"""
num_instances = 2000
num_jobs = 10 # per instance
num_machines = 5 # per instance
num_operations_per_job = 3
mean_duration_per_op_in_epoch = 7
experiment_type = "Homogen"
# experiment_type = "Heterogen"
# experiment_type = "Heterogen_Energy"
# experiment_type = "Homogen_Energy"
# ---------
mixed_objective = False # having a tie breaker for energy-aware optimization
variable_solver_timeout = True
list_solver_max_timeout = [1*solver_max_timeout_in_seconds, 3*solver_max_timeout_in_seconds, 5*solver_max_timeout_in_seconds]
# ------------ Log Directory
# root_log_directory = f"../Logs/GeneralExpv2/{experiment_type}/{location}" # Most of HotCarbon Results
root_log_directory = f"../Logs/GeneralExpv3-VariableTimer/{experiment_type}/{location}" # Variable timeout testing
######## Heterogeneous ########
if experiment_type == "Heterogen" or experiment_type == "Heterogen_Energy":
    # 5 Servers
    if num_machines == 5:
        power = [1, 2, 4, 6, 8] # machines are heterogeneous (power is normalized!)
        duration_coeff = [3, 2, 1, 0.75, 0.5]
    # 10 Servers
    elif num_machines == 10:
        power = [1, 1, 2, 2, 4, 4, 6, 6, 8, 8] # machines are heterogeneous (power is normalized!)
        duration_coeff = [3, 3, 2, 2, 1, 1, 0.75, 0.75, 0.5, 0.5]
    # 15 Servers
    elif num_machines == 15:
        power = [1, 1, 1, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8] # machines are heterogeneous (power is normalized!)
        duration_coeff = [3, 3, 3, 2, 2, 2, 1, 1, 1, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5]
    # 20 Servers
    elif num_machines == 20:
        power = [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8] # machines are heterogeneous (power is normalized!)
        duration_coeff = [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5]
######## Homogeneous ########
elif experiment_type == "Homogen" or experiment_type == "Homogen_Energy":
    power = [1 for _ in range(num_machines)] # machines are homogeneous.
    duration_coeff = [1 for _ in range(num_machines)]
else:
    raise Exception ("Unknown Experiment Type!")
# list_jobs_arrival_epoch = [[0 for _ in range(num_jobs)] for _ in range(num_instances)] # ever job arrives hour0 of the day
list_jobs_arrival_epoch = generate_multiple_job_arrival_epochs(start_hour=0, end_hour=24, epoch_in_minutes=epoch_in_minutes, sample_num=num_instances, num_jobs=num_jobs, seed=42)
with open(f"../Data/JobPool/JobPool_{num_operations_per_job}Ops_MeanOpDur={mean_duration_per_op_in_epoch}_Epoch={epoch_in_minutes}.json", "r") as f:
    job_dict = json.load(f)
list_jobs_data, list_job_ids = generate_multiple_job_schedules(job_dict, num_jobs = num_jobs, num_machines = num_machines, sample_num = num_instances, seed = 42)

"""
Running the flexible jobshop scenario for each instance
"""
def main_parallel(experiment_type, instance_num_start, instance_num_end, version, start_date = pd.to_datetime("2024-01-01").date(), num_instances_per_day = 1, candidate_makespan_slack_coeff = [1, 1.5, 2]):
    # instance_num_start, instance_num_end = 0, num_instances
    # candidate_makespan_slack_coeff = [1, 1.5, 2]
    # candidate_makespan_slack_coeff = [1, 1.5, 2]
    if not mixed_objective:
        log_file_path = f"{root_log_directory}/{num_jobs}J_{num_machines}S_{num_operations_per_job}O_MeanOp={mean_duration_per_op_in_epoch}_v{version}.csv"
    else:
        log_file_path = f"{root_log_directory}/{num_jobs}J_{num_machines}S_{num_operations_per_job}O_MeanOp={mean_duration_per_op_in_epoch}_MixedObjective_v{version}.csv"
    directory = os.path.dirname(log_file_path)
    os.makedirs(directory, exist_ok=True)
    start_time = datetime.now()
    
    manager = Manager()
    start_time = datetime.now()
    selected_date = start_date
    for instance_num in range(instance_num_start, instance_num_end):
        print(f"Running Instance {instance_num}")
        if instance_num % num_instances_per_day == 0 and instance_num != instance_num_start:
            selected_date += pd.Timedelta(days=1)
        carbon_trace_spec_day_per_epoch = generate_instance_carbon_intensity_trace(selected_date = selected_date, location = location)

        global_minimum_makespan, carbon_consumption, energy_consumption, solver_status, servers_status = makespan_minimizer(carbon_trace_spec_day_per_epoch=carbon_trace_spec_day_per_epoch[location],
            jobs_data=list_jobs_data[instance_num],
            jobs_id=list_job_ids[instance_num],
            num_machines=num_machines,
            jobs_arrival_epoch=list_jobs_arrival_epoch[instance_num],
            initial_horizon=initial_horizon,
            solver_max_timeout_in_seconds=solver_max_timeout_in_seconds
        )

        if solver_status != "Success":
            continue

        log_dict_list_shared = manager.list()

        # Log baseline makespan schedule
        log = {
            'ElapsedTime': str(datetime.now() - start_time).split(".")[0], "Datetime": selected_date,
            'Instance': instance_num, 'IsCarbonAware': False,
            'MaxMakeSpan': initial_horizon, 'MinMakeSpan': global_minimum_makespan, 'SlackCoeff': 1, 'Makespan': global_minimum_makespan,
            'CarbonConsumption(g)': round(carbon_consumption, 2), 'EnergyConsumption(kWh)': round(energy_consumption, 2),
            'JobNumber': num_jobs, 'ServerNumber': num_machines, 'OperationsPerJob': num_operations_per_job,
            'ServerPower': power, 'EpochDuration(min)': epoch_in_minutes, 'SolverTimer(min)': solver_max_timeout_in_seconds // 60,
            'JobIndex': list_job_ids[instance_num], 'JobArrivalEpoch': list_jobs_arrival_epoch[instance_num]
        }
        for sid in range(num_machines):
            log[f"Server{sid}"] = servers_status.get(f"Server{sid}", [])
        log_dict_list_shared.append(log)

        candidate_maximum_allowed_makespan = []
        solver_timeouts = []
        for k, slack_coeff in enumerate(candidate_makespan_slack_coeff):
            val = int(math.ceil(slack_coeff * global_minimum_makespan)) # in epochs
            candidate_maximum_allowed_makespan.append(min(val, len(carbon_trace_spec_day_per_epoch[location])))
            if variable_solver_timeout:
                index = -1
                if k < len(list_solver_max_timeout):
                    index = k
                solver_timeouts.append(list_solver_max_timeout[index])
            else:
                solver_timeouts.append(solver_max_timeout_in_seconds)
        # Parallel carbon-aware runs
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    run_objective_aware_task,
                    carbon_trace_spec_day_per_epoch[location], selected_date, instance_num, makespan_limit, candidate_makespan_slack_coeff[i], global_minimum_makespan, start_time,
                    list_jobs_data, list_job_ids, list_jobs_arrival_epoch,
                    num_jobs, num_machines, num_operations_per_job,
                    power, epoch_in_minutes, solver_timeouts[i],
                    log_dict_list_shared,
                    experiment_type
                )
                for i , makespan_limit in enumerate(candidate_maximum_allowed_makespan)
            ]
            for future in as_completed(futures):
                future.result()

        append_to_csv(list(log_dict_list_shared), log_file_path)
def main(experiment_type, start_date = pd.to_datetime("2024-01-01").date(), num_instances_per_day = 1):
    instance_num_start, instance_num_end = 0, num_instances
    candidate_makespan_slack_coeff = [1, 1.5, 2]
    candidate_makespan_slack_coeff = [1, 1.5]
    if not mixed_objective:
        log_file_path = f"{root_log_directory}/{num_jobs}J_{num_machines}S_{num_operations_per_job}O_MeanOp={mean_duration_per_op_in_epoch}.csv"
    else:
        log_file_path = f"{root_log_directory}/{num_jobs}J_{num_machines}S_{num_operations_per_job}O_MeanOp={mean_duration_per_op_in_epoch}_MixedObjective.csv"
    directory = os.path.dirname(log_file_path)
    os.makedirs(directory, exist_ok=True)
    start_time = datetime.now()
    selected_date = start_date
    for instance_num in range(instance_num_start, instance_num_end, 1):
        # run makespan minimizer
        print(f"Running Instance {instance_num}")
        if instance_num % num_instances_per_day == 0 and instance_num != instance_num_start:
            selected_date += pd.Timedelta(days=1)
        carbon_trace_spec_day_per_epoch = generate_instance_carbon_intensity_trace(selected_date = selected_date, location = location)
        global_minimum_makespan, carbon_consumption, energy_consumption, solver_status, servers_status = makespan_minimizer(carbon_trace_spec_day_per_epoch = carbon_trace_spec_day_per_epoch[location],
                                                                    jobs_data = list_jobs_data[instance_num],
                                                                    jobs_id = list_job_ids[instance_num],
                                                                    num_machines = num_machines,
                                                                    jobs_arrival_epoch = list_jobs_arrival_epoch[instance_num], 
                                                                    initial_horizon = initial_horizon,
                                                                    solver_max_timeout_in_seconds = solver_max_timeout_in_seconds)
        if solver_status != "Success":
            continue # Do not run carbon-aware scheduling
        log_dict_list = []
        # log makespan minimizer result
        makespan_log_dict = {'ElapsedTime': str(datetime.now() - start_time).split(".")[0], "Datetime": selected_date,
                            'Instance': instance_num, 'IsCarbonAware': False,
                            'MaxMakeSpan': initial_horizon, 'MinMakeSpan': global_minimum_makespan, 'SlackCoeff': 1, 'Makespan': global_minimum_makespan,
                            'CarbonConsumption(g)': round(carbon_consumption, 2), 'EnergyConsumption(kWh)': round(energy_consumption, 2),
                            'JobNumber': num_jobs, 'ServerNumber': num_machines, 'OperationsPerJob': num_operations_per_job,
                            'ServerPower': power, 'EpochDuration(min)': epoch_in_minutes, 'SolverTimer(min)': solver_max_timeout_in_seconds // 60,
                            'JobIndex': list_job_ids[instance_num], 'JobArrivalEpoch': list_jobs_arrival_epoch[instance_num]
                            }
        for serverid in range(num_machines):
            if f"Server{serverid}" in servers_status.keys():
                makespan_log_dict[f"Server{serverid}"] = servers_status[f"Server{serverid}"]
            else:
                makespan_log_dict[f"Server{serverid}"] = []
        log_dict_list.append(makespan_log_dict)
        # carbon-aware or energy-aware scheduling
        candidate_maximum_allowed_makespan = []
        for slack_coeff in candidate_makespan_slack_coeff:
            val = int(math.ceil(slack_coeff * global_minimum_makespan))
            candidate_maximum_allowed_makespan.append(min(val, len(carbon_trace_spec_day_per_epoch[location])))
        for i, max_allowed_makespan in enumerate(candidate_maximum_allowed_makespan):
            print(f"max allowed makespan is {max_allowed_makespan}")
            curr_timeout = solver_max_timeout_in_seconds
            if variable_solver_timeout:
                index = -1
                if i < len(list_solver_max_timeout):
                    index = i
                curr_timeout = list_solver_max_timeout[index]
            print("TIMEOUT = ", curr_timeout //  60)
            if experiment_type == "Heterogen_Energy" or experiment_type == "Homogen_Energy":
                carbon_consumption, energy_consumption, makespan, solver_status, servers_status = energy_aware_scheduling(carbon_trace_spec_day_per_epoch = carbon_trace_spec_day_per_epoch[location],
                                                                            jobs_data = list_jobs_data[instance_num], 
                                                                            jobs_id = list_job_ids[instance_num],
                                                                            jobs_arrival_epoch = list_jobs_arrival_epoch[instance_num],
                                                                            max_allowed_makespan = max_allowed_makespan,
                                                                            solver_max_timeout_in_seconds = curr_timeout)
            else:    
                carbon_consumption, energy_consumption, makespan, solver_status, servers_status = carbon_aware_scheduling(carbon_trace_spec_day_per_epoch = carbon_trace_spec_day_per_epoch[location],
                                                                            jobs_data = list_jobs_data[instance_num], 
                                                                            jobs_id = list_job_ids[instance_num],
                                                                            jobs_arrival_epoch = list_jobs_arrival_epoch[instance_num],
                                                                            max_allowed_makespan = max_allowed_makespan,
                                                                            solver_max_timeout_in_seconds = curr_timeout)
            if solver_status == "Success":
                objective_aware_log_dict = {'ElapsedTime': str(datetime.now() - start_time).split(".")[0], "Datetime": selected_date,
                            'Instance': instance_num, 'IsCarbonAware': True,
                            'MaxMakeSpan': max_allowed_makespan, 'MinMakeSpan': global_minimum_makespan, 'SlackCoeff': candidate_makespan_slack_coeff[i], 'Makespan': makespan,
                            'CarbonConsumption(g)': round(carbon_consumption, 2), 'EnergyConsumption(kWh)': round(energy_consumption, 2),
                            'JobNumber': num_jobs, 'ServerNumber': num_machines, 'OperationsPerJob': num_operations_per_job,
                            'ServerPower': power, 'EpochDuration(min)': epoch_in_minutes, 'SolverTimer(min)': curr_timeout // 60,
                            'JobIndex': list_job_ids[instance_num], 'JobArrivalEpoch': list_jobs_arrival_epoch[instance_num]
                            }
                for serverid in range(num_machines):
                    if f"Server{serverid}" in servers_status.keys():
                        objective_aware_log_dict[f"Server{serverid}"] = servers_status[f"Server{serverid}"]
                    else:
                        objective_aware_log_dict[f"Server{serverid}"] = []
                log_dict_list.append(objective_aware_log_dict)
            # break
        append_to_csv(data_list=log_dict_list, file_path=log_file_path)
        if (instance_num == 1):
            break
    
# main(experiment_type = experiment_type)
###########
run_ver = 0
candidate_makespan_slack_coeff = [1, 1.5, 2]
# candidate_makespan_slack_coeff = [1]
#-----
start_date = pd.to_datetime("2024-01-01").date()
total_days = 360
num_instances_per_day = 3
num_available_obelix = 8
inst_num_on_each_obelix = (num_instances_per_day * total_days) // num_available_obelix
days_covered_per_obelix = inst_num_on_each_obelix // num_instances_per_day
obelix_start_dates = []
for i in range(num_available_obelix):
    start_day = i * days_covered_per_obelix
    start_dt = start_date + pd.Timedelta(days=start_day)
    print(f"Machine {i}: {inst_num_on_each_obelix} instances, Start Date: {start_dt}")
    obelix_start_dates.append(start_dt)
main_parallel(experiment_type = experiment_type,
              instance_num_start = run_ver * inst_num_on_each_obelix,
              instance_num_end = (run_ver+1) * inst_num_on_each_obelix,
              version = run_ver,
              start_date = obelix_start_dates[run_ver],
              num_instances_per_day = num_instances_per_day,
              candidate_makespan_slack_coeff = candidate_makespan_slack_coeff)