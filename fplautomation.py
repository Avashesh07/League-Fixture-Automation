import pandas as pd
from pyomo.environ import *
from datetime import datetime
from itertools import permutations
from pyomo.opt import SolverStatus, TerminationCondition

# -----------------------------
# Step A: Import Libraries
# -----------------------------
import pandas as pd
from pyomo.environ import *
from datetime import datetime

# -----------------------------
# Step B: Read and Process Data
# -----------------------------
# Load data from Excel
teams_df = pd.read_excel('schedule_data.xlsx', sheet_name='Teams')
grounds_df = pd.read_excel('schedule_data.xlsx', sheet_name='Grounds')

# Extract teams
teams = teams_df['TeamName'].tolist()

# Extract dates and ground availability
grounds_df['Date'] = pd.to_datetime(grounds_df['Date'], format='%d-%m-%Y')  # Adjust format as needed
dates_sorted = sorted(grounds_df['Date'].tolist())

# Extract ground names (excluding 'Date' column)
grounds = list(grounds_df.columns)
grounds.remove('Date')

# Function to determine if a date is a weekend
def is_weekend(date):
    return date.weekday() >= 5  # Saturday=5, Sunday=6

# Add a column to indicate weekend
grounds_df['IsWeekend'] = grounds_df['Date'].apply(is_weekend)

# Initialize a dictionary to hold ground capacities
ground_capacity = {}

for idx, row in grounds_df.iterrows():
    date = row['Date']
    is_weekend = row['IsWeekend']
    for ground in grounds:
        if row[ground] == 1:
            if is_weekend:
                capacity = 3  # 3 matches on weekends
            else:
                capacity = 1  # 1 match on weekdays
            ground_capacity[(ground, date)] = capacity
        else:
            ground_capacity[(ground, date)] = 0  # Ground unavailable

# -----------------------------
# Step E: Generate All Possible Matches
# -----------------------------
# Generate all ordered pairs (home, away) excluding self-matches
matches = [(h, a) for h in teams for a in teams if h != a]

# -----------------------------
# Step F: Initialize Pyomo Model
# -----------------------------
# Initialize the model
model = ConcreteModel()

# Sets
model.Teams = Set(initialize=teams)
model.Dates = Set(initialize=dates_sorted)
model.Grounds = Set(initialize=grounds)
model.Matches = Set(initialize=matches, dimen=2)

# -----------------------------
# Step G: Define Parameters
# -----------------------------
# Ground Availability Parameter
model.GroundAvailability = Param(model.Grounds, model.Dates, initialize=ground_capacity, default=0)

# Determine if a date is weekend
date_weekend = {date: is_weekend(date) for date in dates_sorted}
model.IsWeekend = Param(model.Dates, initialize=date_weekend, within=Binary)

# -----------------------------
# Step H: Define Variables
# -----------------------------
# Binary variable: y[h, a, d, g] = 1 if match (h, a) is scheduled on date d at ground g
model.y = Var(model.Matches, model.Dates, model.Grounds, domain=Binary)

# -----------------------------
# Step I: Define Constraints
# -----------------------------

# 1. Each Match is Scheduled Exactly Once
def match_scheduled_once_rule(model, h, a):
    return sum(model.y[h, a, d, g] for d in model.Dates for g in model.Grounds) == 1
model.MatchScheduledOnce = Constraint(model.Matches, rule=match_scheduled_once_rule)

# 2. Ground Capacity Constraints
def ground_capacity_rule(model, g, d):
    return sum(model.y[h, a, d, g] for h, a in model.Matches) <= model.GroundAvailability[g, d]
model.GroundCapacityConstraint = Constraint(model.Grounds, model.Dates, rule=ground_capacity_rule)

# 3. No Team Plays More Than One Match Per Day
def team_one_match_per_day_rule(model, t, d):
    return sum(model.y[h, a, d, g] for h, a in model.Matches if h == t or a == t for g in model.Grounds) <= 1
model.TeamOneMatchPerDay = Constraint(model.Teams, model.Dates, rule=team_one_match_per_day_rule)

# -----------------------------
# Step J: Define the Objective Function
# -----------------------------
model.Objective = Objective(expr=0, sense=minimize)

# -----------------------------
# Step K: Solve the Model
# -----------------------------
# Choose the solver
solver = SolverFactory('cbc')  # Ensure CBC is installed and accessible

# Solve the model
result = solver.solve(model, tee=True)

# -----------------------------
# Step L: Check and Display the Solution
# -----------------------------
# Check if the solution is feasible
if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
    # Retrieve the schedule
    schedule = []
    for h, a in model.Matches:
        for d in model.Dates:
            for g in model.Grounds:
                if value(model.y[h, a, d, g]) == 1:
                    schedule.append({
                        'HomeTeam': h,
                        'AwayTeam': a,
                        'Date': d.strftime('%d-%m-%Y'),
                        'Ground': g
                    })
    schedule_df = pd.DataFrame(schedule)
    
    # Sort the schedule by date
    schedule_df.sort_values(by='Date', inplace=True)
    
    # Reset index
    schedule_df.reset_index(drop=True, inplace=True)
    
    # Display the schedule
    print(schedule_df)
    
    # Optionally, export to Excel
    schedule_df.to_excel('schedule_output.xlsx', index=False)
else:
    print('No feasible solution found.')
    print('Solver Status:', result.solver.status)
    print('Termination Condition:', result.solver.termination_condition)
