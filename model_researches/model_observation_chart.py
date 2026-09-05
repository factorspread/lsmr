"""
Fan Chart – Two-Regime Running Means + Evolving Distribution Bands
───────────────────────────────────────────────────────────────────
• Running mean 1 : overall cumulative mean of all observations
• Running mean 2 : cumulative mean of observations above threshold (D2)
• D1 bands       : Main distributiom quantile bands, refit at every checkpoint
• D2 bands       : GPD quantile bands if present,   refit at every checkpoint
• Rolling quantile ribbon + y-axis rug + extreme scatter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def plot_obs_charts(dsets:pd.DataFrame, save_path:str):
    for row in dsets.iterrows():

        dset = row[1]
        
        # ── DATA ──────────────────────────────────────────────────────────────────────
        N = 30_000

        body = stats.gamma.rvs(a=2.5, scale=12, size=int(N * 0.83))
        tail = stats.genpareto.rvs(c=0.35, loc=0, scale=22, size=N - int(N * 0.83)) + 55
        data = np.concatenate([body, tail])
        np.random.shuffle(data)

        MAIN_PCT = 85


        size = dset['model']._dep_var.shape[0]
        obs_min = 0
        obs_max = size
        dist_params = {}
        ex_dist_left_params = {}
        ex_dist_right_params = {}
        first_forecast_index = max(dset['ar_order']) - 1
        variance = dset['model'].forecast(horizon=1, start=first_forecast_index, method='simulation').variance.dropna().to_numpy()[obs_min:obs_max-first_forecast_index,0]
        f_values = dset['model'].forecast(horizon=1, start=first_forecast_index, method='simulation').mean.dropna().to_numpy()[obs_min:obs_max-first_forecast_index,0]
        data = dset['model']._dep_var
        #D1 dist
        d1_dist_type = dset['dist_name']
        if d1_dist_type == 'Student-t':
                dist_params['d_f'], dist_params['loc'], dist_params['scale']  = dset['dist'] #t-stat
        elif d1_dist_type == 'Normal':
                dist_params['loc'], dist_params['scale'] = dset['dist'] #Normal
        thr_left = dset['u_threshold_left']
        thr_right = dset['u_threshold_right']

        # Handle No extreme thresholds
        PCTL_LEFT = 0.0015
        PCTL_RIGHT = PCTL_LEFT


        #D2 dist (left, right extremes)
        ex_dist_left_type = 'gpd'
        ex_dist_left = dset['gpd_dist_left']
        ex_dist_right = dset['gpd_dist_right']

        # Check if extreme distributions exist
        has_left_extreme = ex_dist_left if not pd.isna(ex_dist_left) else None
        has_right_extreme = ex_dist_right if not pd.isna(ex_dist_right) else None

        if has_left_extreme:
            ex_dist_left_params['c'], ex_dist_left_params['loc'], ex_dist_left_params['scale'] = ex_dist_left.args[0], ex_dist_left.kwds['loc'], ex_dist_left.kwds['scale']

        if has_right_extreme:
            ex_dist_right_params['c'], ex_dist_right_params['loc'], ex_dist_right_params['scale'] = ex_dist_right.args[0], ex_dist_right.kwds['loc'], ex_dist_right.kwds['scale']

        data = data.iloc[obs_min + first_forecast_index:obs_max]

        N = f_values.shape[0]

        obs = np.arange(1, N + 1)

        # ── PER-CHECKPOINT DISTRIBUTION FITS ─────────────────────────────────────────
        MIN_N, STEP = 1, 20

        if d1_dist_type == 'Student-t':
            if pd.isna(thr_left):
                thr_pctl_left = PCTL_LEFT
            else: 
                thr_pctl_left=stats.t.pdf(thr_left, df=dist_params['d_f'], loc=dist_params['loc'], scale=dist_params['scale'])
            if pd.isna(thr_right):
                thr_pctl_right = PCTL_RIGHT
            else: 
                thr_pctl_right = stats.t.pdf(thr_right, df=dist_params['d_f'], loc=dist_params['loc'], scale=dist_params['scale'])

        checkpoints = np.arange(MIN_N, N + 1, STEP)
        Q_PAIRS_D1     = [(thr_pctl_left, 1-thr_pctl_right), (0.10, 0.90)]
        Q_PAIRS_D2     = [(0., 0.95)]

        d1_lo   = np.full((len(checkpoints), 3), np.nan)
        d1_hi   = np.full((len(checkpoints), 3), np.nan)
        ex_lo_left   = np.full((len(checkpoints), 3), np.nan)
        ex_lo_right   = np.full((len(checkpoints), 3), np.nan)
        ex_hi_left   = np.full((len(checkpoints), 3), np.nan)
        ex_hi_right   = np.full((len(checkpoints), 3), np.nan)
        ex_left_mean = np.full((len(checkpoints), 3), np.nan)
        ex_right_mean = np.full((len(checkpoints), 3), np.nan)
        d1_thr_hi = np.full(len(checkpoints), np.nan)
        d1_thr_lo = np.full(len(checkpoints), np.nan)

        for k, cp in enumerate(checkpoints):
                mean       = f_values[cp]
                std_dev    = variance[cp]**(1/2) 

                for j, (ql, qh) in enumerate(Q_PAIRS_D1):
                        if d1_dist_type == 'Student-t':
                                d1_lo[k, j] = mean + stats.t.ppf(ql, df=dist_params['d_f'], loc=dist_params['loc'], scale=dist_params['scale']) * std_dev
                                d1_hi[k, j] = mean + stats.t.ppf(qh, df=dist_params['d_f'], loc=dist_params['loc'], scale=dist_params['scale']) * std_dev
                        elif d1_dist_type == 'Normal':
                                d1_lo[k, j] = mean + stats.norm.ppf(ql, loc=dist_params['loc'], scale=dist_params['scale']) * std_dev
                                d1_hi[k, j] = mean + stats.norm.ppf(qh, loc=dist_params['loc'], scale=dist_params['scale']) * std_dev

                for j, (ql, qh) in enumerate(Q_PAIRS_D2):
                        if has_left_extreme:
                            ex_lo_left[k, j] = mean + (stats.genpareto.ppf(ql, c=ex_dist_left_params['c'], loc=ex_dist_left_params['loc'], scale=ex_dist_left_params['scale']) * std_dev) * (-1)
                            ex_hi_left[k, j] = mean + (stats.genpareto.ppf(qh, ex_dist_left_params['c'], loc=ex_dist_left_params['loc'], scale=ex_dist_left_params['scale']) * std_dev) * (-1)
                            ex_left_mean[k, j] = mean + (stats.genpareto.ppf(0.5, c=ex_dist_left_params['c'], loc=ex_dist_left_params['loc'], scale=ex_dist_left_params['scale']) * std_dev) * (-1)
                        
                        if has_right_extreme:
                            ex_lo_right[k, j] = mean + stats.genpareto.ppf(ql, c=ex_dist_right_params['c'], loc=ex_dist_right_params['loc'], scale=ex_dist_right_params['scale']) * std_dev
                            ex_hi_right[k, j] = mean + stats.genpareto.ppf(qh, c=ex_dist_right_params['c'], loc=ex_dist_right_params['loc'], scale=ex_dist_right_params['scale']) * std_dev
                            ex_right_mean[k, j] = mean + stats.genpareto.ppf(0.5, c=ex_dist_right_params['c'], loc=ex_dist_right_params['loc'], scale=ex_dist_right_params['scale']) * std_dev

        def interp_band(arr_cp):
            return np.interp(obs, checkpoints, arr_cp)

        # ── ROLLING QUANTILE RIBBON ───────────────────────────────────────────────────
        WIN  = 20
        half = WIN // 2
        q10  = np.array([np.percentile(data[max(0, i - half):i + half], 10) for i in range(N)])
        q25  = np.array([np.percentile(data[max(0, i - half):i + half], 25) for i in range(N)])
        q50  = np.array([np.percentile(data[max(0, i - half):i + half], 50) for i in range(N)])
        q75  = np.array([np.percentile(data[max(0, i - half):i + half], 75) for i in range(N)])
        q90  = np.array([np.percentile(data[max(0, i - half):i + half], 90) for i in range(N)])
        qmax = np.array([np.percentile(data[max(0, i - half):i + half], (1-thr_pctl_right)*100) for i in range(N)])
        qmin = np.array([np.percentile(data[max(0, i - half):i + half], thr_pctl_left*100) for i in range(N)])

        # ── EXTREME OBSERVATIONS ──────────────────────────────────────────────────────
        p_thr_lo_left       = np.percentile(data, thr_pctl_left*100)
        p_thr_lo_right       = np.percentile(data, thr_pctl_right*100)
        p_thr_hi_left       = np.percentile(data, (1-thr_pctl_left)*100)
        p_thr_hi_right       = np.percentile(data, (1-thr_pctl_right)*100)
        mask_ext_hi_left  = data > p_thr_hi_left
        mask_ext_hi_right  = data > p_thr_hi_right
        mask_ext_lo_left  = data < p_thr_lo_left
        mask_ext_lo_right  = data < p_thr_lo_right
        obs_ext_left   = obs[mask_ext_lo_left]
        obs_ext_right   = obs[mask_ext_hi_right]
        data_ext_lo_left  = data[mask_ext_lo_left]
        data_ext_lo_right  = data[mask_ext_lo_right]
        data_ext_hi_left  = data[mask_ext_hi_left]
        data_ext_hi_right  = data[mask_ext_hi_right]

        # ── PALETTE – WHITE BACKGROUND ────────────────────────────────────────────────
        BG        = "#ffffff"
        LABEL     = "#1e293b"
        MUTED     = "#64748b"
        GRID_COL  = "#e2e8f0"

        C_D1_EDGE  = "#0d9488"
        D1_FILLS   = ["#ccfbf1", "#99f6e4", "#5eead4"]
        D1_ALPHA   = [0.9, 0.9, 0.9]

        C_D2_EDGE  = "#d97706"
        D2_FILLS   = ["#fef3c7", "#fde68a", "#fcd34d"]
        D2_ALPHA   = [0.9, 0.9, 0.9]

        C_THR   = "#94a3b8"
        C_RIB   = "#3b82f6"
        C_M1    = "#16a34a"
        C_M2    = "#ea580c"
        C_EXT   = "#dc2626"
        C_RUG   = "#475569"

        plt.rcParams.update({
            "figure.facecolor":  BG,
            "axes.facecolor":    BG,
            "axes.edgecolor":    "#cbd5e1",
            "axes.labelcolor":   LABEL,
            "xtick.color":       MUTED,
            "ytick.color":       MUTED,
            "xtick.labelsize":   9,
            "ytick.labelsize":   9,
            "text.color":        LABEL,
            "grid.color":        GRID_COL,
            "grid.linewidth":    0.8,
            "font.family":       "DejaVu Sans",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

        # ── FIGURE ────────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(20, 9))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10)

        y_ceil_lo   = np.percentile(data, thr_pctl_left*100)
        y_ceil_high  = np.percentile(data, thr_pctl_right*100)

        # ── D1 BANDS ──────────────────────────────────────────────────────────────────
        for j in range(2):
            lo = interp_band(d1_lo[:, j])
            hi = interp_band(d1_hi[:, j])
            ax.fill_between(obs, lo, hi, color=D1_FILLS[j], alpha=D1_ALPHA[j], zorder=2)
        ax.plot(obs, interp_band(d1_hi[:, 0]),
                color=C_D1_EDGE, lw=0.9, alpha=0.7, zorder=3)
        ax.plot(obs, interp_band(d1_lo[:, 0]),
                color=C_D1_EDGE, lw=0.9, alpha=0.7, zorder=3)

        # ── D2 BANDS ──────────────────────────────────────────────────────────────────
        # Only plot extreme distributions if they exist
        if has_right_extreme:
            hi_right = interp_band(ex_hi_right[:, 0])
            ax.fill_between(obs, interp_band(d1_hi[:, 0]), hi_right, color=D2_FILLS[2], alpha=D2_ALPHA[2], zorder=2)

        if has_left_extreme:
            hi_left = interp_band(ex_hi_left[:, 0])
            ax.fill_between(obs, interp_band(d1_lo[:, 0]), hi_left, color=D2_FILLS[2], alpha=D2_ALPHA[2], zorder=2)

        if has_right_extreme:
            lo = interp_band(ex_lo_right[:, 0])

        # ── ROLLING QUANTILE RIBBON ───────────────────────────────────────────────────
        ax.fill_between(obs, q10, q90, color=C_RUG, alpha=0.17, zorder=5)
        ax.fill_between(obs, qmin, qmax, color=C_RIB, alpha=0.17, zorder=6)
        ax.plot(obs, q50, color=C_RIB, lw=1.8, alpha=0.9, zorder=7)

        # ── EXTREME SCATTER > pmax | < pmin ─────────────────────────────────────────────────────
        #if has_right_extreme:
        ax.scatter(obs_ext_right, data_ext_hi_right, s=7, alpha=0.45, color=C_EXT,
                    linewidths=0, rasterized=True, zorder=9)

        #if has_left_extreme:
        ax.scatter(obs_ext_left, data_ext_lo_left, s=7, alpha=0.45, color=C_EXT,
                    linewidths=0, rasterized=True, zorder=9)

        # ── RUNNING MEANS ─────────────────────────────────────────────────────────────
        if has_left_extreme:
            ax.plot(obs, interp_band(ex_left_mean[:, 0]),  color=C_M2, lw=1, zorder=10,
                    label=f"Mean 2 – D₂")
        if has_right_extreme:
            ax.plot(obs, interp_band(ex_right_mean[:, 0]),  color=C_M2, lw=1, zorder=10,
                    label=f"Mean 2 – D₂")
            
        # ── AXES ──────────────────────────────────────────────────────────────────────
        ax.set_xlim(0, N)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        ax.set_xlabel("Observation index", fontsize=11, labelpad=8)
        ax.set_ylabel("Value", fontsize=11, labelpad=8)
        ax.grid(True, axis="y", zorder=0)
        ax.grid(True, axis="x", zorder=0, alpha=0.6)

        # ── LEGEND ────────────────────────────────────────────────────────────────────
        handles = [
            # D1
            mpatches.Patch(color=D1_FILLS[0], ec=C_D1_EDGE, lw=0.8, label=f"D₁  {d1_dist_type}  {(thr_pctl_left*100):.2f}-{((1-thr_pctl_right)*100):.2f} %"),
            mpatches.Patch(color=D1_FILLS[1], ec=C_D1_EDGE, lw=0.8, label=f"D₁  {d1_dist_type}  10–90 %"),
            # Empirical ribbon
            mpatches.Patch(color=C_RIB, alpha=0.12, label="Empirical p10–p90"),
            mpatches.Patch(color=C_RIB, alpha=0.28, label=f"Empirical p{thr_pctl_left*100:.2f}–p{(1-thr_pctl_right)*100:.2f}"),
            Line2D([0],[0], color=C_RIB, lw=2.0,   label="Empirical median (p50)"),
        ]

        handles.append(Line2D([0],[0], color=C_EXT, lw=0, marker="o", ms=5, alpha=0.7,
                                label=f"Obs > p{(1-thr_pctl_right)*100:.2f} < p{(thr_pctl_left)*100:.2f}  ({mask_ext_hi_right.sum():.0f} pts)"))

        # Only add D2 and extreme-related legend items if distributions exist
        if has_left_extreme or has_right_extreme:
            handles.append(mpatches.Patch(color=D2_FILLS[0], ec=C_D2_EDGE, lw=0.8, label=f"D₂ GPD  0–95 %"))

        if has_left_extreme or has_right_extreme:
            handles.append(Line2D([0],[0], color=C_M2, lw=2.4, label=f"Mean D₂"))

        leg = ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=9,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.97,
            ncol=2,
            columnspacing=1.6,
            handlelength=1.8,
            borderpad=0.9,
            labelspacing=0.5,
        )
        leg.get_frame().set_linewidth(0.8)

        # ── TITLE ─────────────────────────────────────────────────────────────────────
        if has_left_extreme or has_right_extreme:
            ext = f"D₁  {d1_dist_type} (teal) " + "D₂ GPD (amber)" 
        else: 
            ext = ""
        ax.set_title(
            f"{dset['m_p']}  ·  N = {N:,}  · {ext}  Rolling empirical quantiles + extreme observations (< p{(thr_pctl_left*100):.2f}) (> p{((1-thr_pctl_right)*100):.2f})",
            color=LABEL, fontsize=11, fontweight="bold", pad=14, loc="left",
        )

        plt.savefig(f"{save_path}/{dset['m_p']}.png")

        plt.tight_layout()
        plt.show()