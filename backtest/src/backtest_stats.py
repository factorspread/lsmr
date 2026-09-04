
import threading
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hftbacktest.stats import LinearAssetRecord
from math import ceil

class BackTestStats():
    """
    BackTestStats object holds all the stats generated for a markets pairs combination
    """
    def __init__(self, npz_path:str):
        self.source_npz = npz_path
        self.m1_LinearAssetRecord = None
        self.m2_LinearAssetRecord = None
        self.pnl = None
        self.pnl_out_path = None
        self.sharpe = None
        self.sortino = None
        self.graph = None
        self.graph_out_path = None


class StatsLstmr():
    """
    Produce the strategies statistics and charts

    Args:
        - strategy_results: Dictionary with the markets pairs combination as keys
                            and path to the NPZ files from the Hftbacktest recorder
        - num_workers: Number of workers running in parallel (default : 8). 
                       Files I/O is multithreaded while computations as charts generation are multiprocessed
        - equity: The equity value in pairs quote currency applied to each M1 and M2
    """
    def __init__(self, strategy_results:dict, equity:float , num_workers:int=8):
        self.EQUITY = equity

        self.strategy_results = {k:BackTestStats(v) for k,v in strategy_results.items()}

        #Prepare batches
        self.batches = []
        procs_idx = [k for k in self.strategy_results.keys()]
        for i in range(ceil(len(strategy_results) / num_workers)):
            self.batches.append(procs_idx[i * num_workers: (i + 1) * num_workers])

        #Load all the provided results
        self._multi_imports()


    def _import_results(self, k:str):
        """
        Load backtest results through single NPZ file from the Hftbacktest recorder.
        
        Args:
            - k: The markets pairs combination as specified in `strategy_results`
        """
        
        print(f"Loading {k}...")

        current = self.strategy_results[k]
        res_npz = np.load(current.source_npz)
        current.m1_LinearAssetRecord, current.m2_LinearAssetRecord = \
            LinearAssetRecord(res_npz["0"]).stats(book_size=self.EQUITY),\
            LinearAssetRecord(res_npz["1"]).stats(book_size=self.EQUITY)

    def _multi_imports(self):
        threads = []

        print("Importing backtest results...")

        #Run in batches where each batch <= `num_workers`
        for b in self.batches:        
            for k in b:
                t = threading.Thread(
                    target=self._import_results,
                    name=f"Thread {k}",
                    daemon=True,
                    args=(k,)
                )
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

        print("Done")

    def _stats_genrator(self, k:str, q:multiprocessing.Queue):
        self.curr = self.strategy_results[k]

        self.curr.pnl = self.generate_pnl(self.curr)
        self.curr.pnl_out_path = self.save_pnl(self.curr, k, self.base_path)
        self.curr.sharpe = self.calc_sharpe(self.curr)
        self.curr.sortino = self.calc_sortino(self.curr)
        self.curr.graph = self.generate_chart(self.curr, k)
        self.curr.graph_out_path = self.save_chart(self.curr, k, self.base_path)

        q.put({k : self.curr})        

    def produce_stats(self, base_directory:str):
        """
        Generate the statistics and graphs in multiprocessed

        Args:
           - base_directory : The path to the base directory to save into
        """

        #Standardize output path to avoid undesidred output paths
        self.base_path = base_directory
        if self.base_path[-1] == "/":
            self.base_path = self.base_path[:-1]

        procs = []
        q = multiprocessing.Queue()

        #Run in batches where each batch <= `num_workers`
        for b in self.batches:
            batch_procs = []   
            for k in b:
                p = multiprocessing.Process(
                    target=self._stats_genrator,
                    name=f"Process {k}",
                    daemon=True,
                    args=(k,q)
                )
                p.start()
                procs.append(p)
                batch_procs.append(p)
            for _ in batch_procs:
                self.strategy_results.update(q.get())

        for p in procs:
            p.join()


    def generate_pnl(self, curr:BackTestStats):
        """
        Generate the cumulative PnL of the strategy over time.

        Args:
            - curr: Current BackTestStats object
        """
        pnl = pd.DataFrame(columns=['pnl', 'pnl_pct'])

        m1_rec, m2_rec = [curr.m1_LinearAssetRecord.entire.to_pandas(), curr.m2_LinearAssetRecord.entire.to_pandas()]

        #PnL is equal to the sum of the fees adjusted portfolio equities on M1 and M2
        m1_net_eq = m1_rec['equity_wo_fee'] -  m1_rec['fee']
        m2_net_eq = m2_rec['equity_wo_fee'] - m2_rec['fee']
        pnl['pnl'] = (m1_net_eq + m2_net_eq) #Quote currency PnL
        pnl['pnl_pct'] = ((pnl['pnl'] / (self.EQUITY * 2)) * 100.0) #PnL as percentage of initial equity

        return pnl
    
    def save_pnl(self, curr:BackTestStats, k:str, base_directory:str):
        """
        Saves the PnL results in CSV at `base_directory`/pnl/`k`.csv
        
        Args:
           - curr: Current BackTestStats object            
           - k: The markets pairs combination as specified in `strategy_results`
           - base_directory : The path to the base directory to save into
        """

        pnl_out_path = f"{base_directory}/pnl/{k}.csv"
        curr.pnl.to_csv(pnl_out_path)

        print(f"Saved {pnl_out_path}")

        return pnl_out_path


    def calc_sharpe(self, curr:BackTestStats):
        """
        Calculate the strategy Shapre ratio.
        Requires PnL to be generated first.

        Args:
            - curr: Current BackTestStats object
        """
        pnl_pct = curr.pnl['pnl_pct']

        sharpe = pnl_pct.iloc[-1]/np.std(pnl_pct)

        return sharpe

    def calc_sortino(self, curr:BackTestStats, mar:float=0.0):
        """
        Calculate the strategy Sortino ratio.
        Requires PnL to be generated first.

        Args:
            - curr: Current BackTestStats object
            - mar: The minimum acceptable percentage returns in decimal (default 0)     
        """
        pnl_pct = curr.pnl['pnl_pct']

        sortino = pnl_pct.iloc[-1]/np.std(pnl_pct[pnl_pct < mar])

        return sortino

    def generate_chart(self, curr:BackTestStats, k:str):
        """
        Generate the chart to show the PnL moves across the backtesting window and display performance statistics

        Args:
            - curr: Current BackTestStats object
        """
        pnl_pct = curr.pnl['pnl_pct'].to_numpy()
        time = [i for i in range(pnl_pct.shape[0])]

        # Running max / drawdown for shading
        running_max = np.maximum.accumulate(pnl_pct)
        drawdown = pnl_pct - running_max

        # ---------------------------------------------------------
        # 2. Plot
        # ---------------------------------------------------------
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        line_color = "#1f6feb"
        fill_pos = "#d6e6ff"
        fill_neg = "#ffe0e0"

        ax.plot(time, pnl_pct, color=line_color, linewidth=1.8, zorder=3, label="Cumulative PnL")

        # Fill above/below zero
        ax.fill_between(time, pnl_pct, 0, where=(pnl_pct >= 0), color=fill_pos, alpha=0.6, zorder=1)
        ax.fill_between(time, pnl_pct, 0, where=(pnl_pct < 0), color=fill_neg, alpha=0.6, zorder=1)

        # Zero line
        ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=2)

        # Mark peak and trough
        peak_idx = int(np.argmax(pnl_pct))
        trough_idx = int(np.argmin(pnl_pct))
        ax.scatter(time[peak_idx], pnl_pct[peak_idx], color="#2ecc71", s=45, zorder=4, edgecolor="white", linewidth=0.8)
        ax.annotate(f"Peak: %{pnl_pct[peak_idx]:,.0f}",
                    xy=(time[peak_idx], pnl_pct[peak_idx]),
                    xytext=(10, 12), textcoords="offset points",
                    fontsize=9, color="#2ecc71", fontweight="bold")

        ax.scatter(time[trough_idx], pnl_pct[trough_idx], color="#e74c3c", s=45, zorder=4, edgecolor="white", linewidth=0.8)
        ax.annotate(f"Trough: %{pnl_pct[trough_idx]:,.0f}",
                    xy=(time[trough_idx], pnl_pct[trough_idx]),
                    xytext=(10, -18), textcoords="offset points",
                    fontsize=9, color="#e74c3c", fontweight="bold")

        # ---------------------------------------------------------
        # 3. Styling
        # ---------------------------------------------------------
        ax.set_title(f"{k} Cumulative PnL", fontsize=16, fontweight="bold", pad=16, color="#1a1a1a")
        ax.set_xlabel("Time (t=5min)", fontsize=11, color="#444444")
        ax.set_ylabel("Cumulative PnL (%)", fontsize=11, color="#444444")


        fig.autofmt_xdate(rotation=30)

        ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5, color="#e5e5e5", zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#cccccc")

        # Summary stats box
        total_pnl = pnl_pct[-1]
        max_dd = drawdown.min()
        win_rate = (np.sum((np.diff(pnl_pct) > 0)) / pnl_pct.shape[0]) * 100
        stats_text = (f"Total PnL: {total_pnl:,.0f}%\n"
                    f"Sharpe ratio :{curr.sharpe:,.2f}\n"
                    f"Sortino ratio :{curr.sortino:,.2f}\n"
                    f"Max Drawdown: {max_dd:,.0f}%\n"
                    f"Win Rate: {win_rate:.1f}%")
        ax.text(0.15, 1., stats_text, transform=ax.transAxes,
                fontsize=9.5, va="top", ha="right", color="#333333",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dddddd", alpha=0.9))

        ax.legend(loc="lower right", frameon=False, fontsize=9)

        plt.tight_layout()

        return [fig, ax]

    def save_chart(self, curr:BackTestStats, k:str, base_directory:str):
        """
        Saves the graph as a PNG image at `base_directory`/graphs/`k`.png
        
        Args:
           - curr: Current BackTestStats object
           - k: The markets pairs combination as specified in `strategy_results`
           - base_directory : The path to the base directory to save into
        """

        graph_out_path = f"{base_directory}/graphs/{k}.png"
        curr.graph[0].savefig(
            graph_out_path, 
            dpi=150, 
            bbox_inches="tight"
            )

        print(f"Saved {graph_out_path}")

        return graph_out_path
