#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Matilde Bureau, Gaston Ravanas
Basilisk Plotting Script for Reports

- Individual Plots: 4-panel analysis for 'dipole' and 'image' cases.
- Layout: [0,0] d_sep, [0,1] Gamma, [1,0] Coordinates, [1,1] Velocities.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob
import shutil
from collections import defaultdict

# --- Report-Quality Visual Styling ---
plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.labelsize': 22,      # X and Y axis labels
    'axes.titlesize': 22,      # Subplot titles
    'xtick.labelsize': 18,     # X-tick numbers
    'ytick.labelsize': 18,     # Y-tick numbers
    'legend.fontsize': 16,     # Legend text
    'grid.alpha': 0.5,
    'lines.linewidth': 3,      # Thicker lines for visibility
    'figure.titlesize': 24
})

# ==========================================
# 1. Helper & Physics Functions
# ==========================================

def find_column(df_cols, aliases):
    """Finds the first matching column name from a list of aliases."""
    for alias in aliases:
        for col in df_cols:
            if alias.lower() == col.lower():
                return col
    return None

def extract_param_value(filename, param_type):
    """Extracts numerical values for parameters (Re, G2, etc.) from filename."""
    pattern = rf"{param_type}_([0-9]+\.?[0-9]*)"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        try: return float(match.group(1))
        except ValueError: return 0.0
    return None

def detect_case(filename):
    """Categorizes the simulation based on filename keywords."""
    fname_lower = os.path.basename(filename).lower()
    for case in ['corot', 'dipole', 'single', 'image']:
        if case in fname_lower:
            return case
    return 'other'

def compute_physics(df):
    """Computes derived quantities: Peak Velocities and Mean Vorticity."""
    col_t = find_column(df.columns, ['t', 'time'])
    col_circ = find_column(df.columns, ['circ', 'gamma', 'G'])
    col_a1 = find_column(df.columns, ['a1', 'radius1', 'a', 'radius'])
    
    col_xmax = find_column(df.columns, ['xc', 'xmax', 'x_max'])
    col_ymax = find_column(df.columns, ['yc', 'ymax', 'y_max'])
    col_xmin = find_column(df.columns, ['xmin', 'x_min'])
    col_ymin = find_column(df.columns, ['ymin', 'y_min'])

    results = {'coords': {}, 'vels': {}}
    if col_t is not None:
        t = df[col_t].values
        if col_xmax and col_ymax:
            results['coords']['xmax'], results['coords']['ymax'] = df[col_xmax], df[col_ymax]
            results['vels']['ux_max'] = np.gradient(df[col_xmax].values, t)
            results['vels']['uy_max'] = np.gradient(df[col_ymax].values, t)
        if col_xmin and col_ymin:
            results['coords']['xmin'], results['coords']['ymin'] = df[col_xmin], df[col_ymin]
            results['vels']['ux_min'] = np.gradient(df[col_xmin].values, t)
            results['vels']['uy_min'] = np.gradient(df[col_ymin].values, t)
    
    if col_circ and col_a1:
        results['omega_mean'] = df[col_circ] / (np.pi * df[col_a1]**2)
    
    return results

# ==========================================
# 2. Plotting Logic
# ==========================================

def plot_individual_file(filepath):
    """Generates specific diagnostic plots based on the simulation case."""
    case_type = detect_case(filepath)
    df = pd.read_csv(filepath, sep='\s+', comment='#')
    if df.empty: return None

    col_t = find_column(df.columns, ['t', 'time'])
    t = df[col_t]
    physics = compute_physics(df)
    
    if case_type in ['dipole', 'image']:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        # No suptitle as requested
        
        # 1. Separation [0, 0] - No legend for single curve
        col_dsep = find_column(df.columns, ['d_sep', 'separation', 'dist'])
        if col_dsep:
            axs[0, 0].plot(t, df[col_dsep], color='black')
            axs[0, 0].set_ylabel(r'$d_{sep}$ (m)')

        # 2. Circulation [0, 1] - No legend for single curve
        col_circ = find_column(df.columns, ['circ', 'gamma', 'G'])
        if col_circ:
            axs[0, 1].plot(t, df[col_circ], color='tab:red')
            axs[0, 1].set_ylabel(r'$\Gamma$ (m$^2$/s)')

        # 3. Coordinates [1, 0] - Legend required
        c = physics['coords']
        if 'xmax' in c:
            axs[1, 0].plot(t, c['xmax'], label='$x_{max}$', color='tab:green')
            axs[1, 0].plot(t, c['ymax'], label='$y_{max}$', color='tab:green', linestyle='--')
        if 'xmin' in c:
            axs[1, 0].plot(t, c['xmin'], label='$x_{min}$', color='tab:brown')
            axs[1, 0].plot(t, c['ymin'], label='$y_{min}$', color='tab:brown', linestyle='--')
        axs[1, 0].set_ylabel('Coordinates (m)')
        axs[1, 0].legend(ncol=2, frameon=True)

        # 4. Velocities [1, 1] - Legend required
        v = physics['vels']
        if 'ux_max' in v:
            axs[1, 1].plot(t, v['ux_max'], label='$u_{max}$', color='tab:green')
            axs[1, 1].plot(t, v['uy_max'], label='$v_{max}$', color='tab:green', linestyle='--')
        if 'ux_min' in v:
            axs[1, 1].plot(t, v['ux_min'], label='$u_{min}$', color='tab:brown')
            axs[1, 1].plot(t, v['uy_min'], label='$v_{min}$', color='tab:brown', linestyle='--')
        axs[1, 1].set_ylabel('Velocity (m/s)')
        axs[1, 1].legend(ncol=2, frameon=True)

    else:
        # Standard 3-panel plot for other cases
        fig, axs = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
        
        col_a1 = find_column(df.columns, ['a1', 'radius1', 'a'])
        if col_a1:
            axs[0].plot(t, df[col_a1], color='tab:blue')
            axs[0].set_ylabel('$a$ (m)')
        
        v = physics['vels']
        if 'ux_max' in v:
            axs[1].plot(t, v['ux_max'], label='$u$')
            axs[1].plot(t, v['uy_max'], label='$v$', linestyle='--')
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend()

        if 'omega_mean' in physics:
            axs[2].plot(t, physics['omega_mean'], color='tab:purple')
            axs[2].set_ylabel(r'$\bar{\omega}$ (s$^{-1}$)')

    for ax in fig.axes:
        ax.set_xlabel('t (s)')
        ax.grid(True)
        # Tick parameters for even better readability
        ax.tick_params(axis='both', which='major', length=7, width=2)

    out_name = os.path.basename(filepath).replace('.out', '_analysis.pdf')
    plt.savefig(os.path.join(os.path.dirname(filepath), out_name))
    plt.close()

def generate_comparison_plot(files, target_param, full_path, case_type):
    """Comparative study: Plots influence of a parameter with high readability."""
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(files)))
    
    for i, f_info in enumerate(files):
        df = pd.read_csv(f_info['path'], sep='\s+', comment='#')
        col_t = find_column(df.columns, ['t', 'time'])
        t = df[col_t]
        physics = compute_physics(df)
        label = f"{target_param} = {f_info[target_param]}"
        
        col_dsep = find_column(df.columns, ['d_sep', 'separation'])
        if col_dsep:
            axs[0].plot(t, df[col_dsep], color=colors[i], label=label)
            axs[0].set_ylabel(r'$d_{sep}$ (m)')
        
        if 'omega_mean' in physics:
            axs[1].plot(t, physics['omega_mean'], color=colors[i])
            axs[1].set_ylabel(r'$\bar{\omega}$ (s$^{-1}$)')

    axs[0].legend(title=f"Varying {target_param}", frameon=True)
    for ax in axs:
        ax.set_xlabel('t (s)')
        ax.grid(True)
        ax.tick_params(axis='both', which='major', length=7, width=2)
    
    plt.savefig(full_path)
    plt.close()

# ==========================================
# 3. Main Automation Workflow
# ==========================================

if __name__ == "__main__":
    raw_files = glob.glob("*.out")
    if not raw_files:
        print("No .out files found in directory.")
    else:
        all_file_data = []
        for f in raw_files:
            case = detect_case(f)
            os.makedirs(case, exist_ok=True)
            new_path = os.path.join(case, f)
            shutil.move(f, new_path)
            all_file_data.append({
                'path': new_path, 'case': case,
                'Re': extract_param_value(f, 'Re'),
                'G2': extract_param_value(f, 'G2'),
                'a2': extract_param_value(f, 'a2')
            })

        for info in all_file_data:
            plot_individual_file(info['path'])

        # Multi-file comparison study logic remains same but inherits visual upgrades
        for param in ["Re", "G2", "a2"]:
            groups = defaultdict(list)
            fixed_params = [p for p in ["case", "Re", "G2", "a2"] if p != param]
            for info in all_file_data:
                if info[param] is not None:
                    key = tuple(info[p] for p in fixed_params)
                    groups[key].append(info)
            for key, group_files in groups.items():
                if len(group_files) > 1:
                    group_files.sort(key=lambda x: x[param])
                    c_type = key[0]
                    desc = "_".join([f"{fixed_params[j]}{key[j]}" for j in range(1, len(fixed_params))])
                    out_path = os.path.join(c_type, f"compare_{param}_{desc}.pdf")
                    generate_comparison_plot(group_files, param, out_path, c_type)

        print("Workflow Complete.")