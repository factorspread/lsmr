import pandas as pd
import numpy as np
import logging
import multiprocessing
from hftbacktest.data.validation import validate_event_order
from math import ceil
from hftbacktest import HashMapMarketDepthBacktest, Recorder
from utils import l2_convert, get_fees, find_closest_across_dataframes
from hbtasset import asset_const
from arch import arch_model, univariate
from datetime import datetime, timedelta

from src.backtest_stats import *


#Mute unnecessary warnings
import warnings
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



class LiquidStackingMeanReversion():

    def __init__(self, **kwargs):

        if not kwargs:
            #Define the backtester parameters and set default values
            self.FIELD_TIMESTAMP = "time_exchange"
            self.TRUNC_TIME_DELTA = "50ms"
            self.LOOKAHEAD_TOLERANCE = 0.1
            self.TIMESTAMP_PRECISION = "ns"
            self.MAIN_DISTRIBUTION = "t"
            self.AR_MODEL_MEAN_TYPE = "AR"
            self.AR_MODEL_TYPE = "FIGARCH"
            self.AR_LAGS_FIELD = "ar_order"
            self.MIN_SAMPLE_SIZE = 1_000
            self.MARKET_FEES_JSON_PATH = "./exchanges_fees.json"
            self.EQUITY = 1_000
            self.MAX_EQ_LOSS = 0.5
            self.MAX_LOSS_PROB = 1/1e4
            self.STEP_TIME_INTEVAL = 300
            self.MIN_PROBABILITY_ENTRY_SIGNAL = 0.6
            self.MAX_PROBABILITY_STOP_LOSS_SIGNAL = 0.1
            self.OUTPUT_BASEPATH = "./results"
        else:
            self.__dict__ = kwargs

        self._init_logger()

    def _init_logger(self):
        #Instantiate logger
        self.logger = logging.getLogger("LSMR")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @staticmethod
    def _second_to_precision(seconds:float, time_precision:str):
        """
        Convert a value in seconds to its value in the time precision.
        
        Args:
            - seconds : The value in seconds
            - time_precision : The time precision unit (e.g. `ns` or for nanoseconds)
        """

        precisions = {
            "ns" : 1e9
        }

        return seconds * precisions.get(time_precision)

    @staticmethod
    def _bba_from_l2(row:pd.DataFrame):
        """ 
        Returns the best bid and ask with the associated volume
        To be called from DataFrame.
        """
        asks = pd.DataFrame([(r['price'],r['size']) for r in row['asks']], columns=['price', 'size'])
        ba = asks[asks['price'] == asks['price'].min()]

        bids = pd.DataFrame([(r['price'],r['size']) for r in row['bids']], columns=['price', 'size'])
        bb = bids[bids['price'] == bids['price'].max()]
        
        return pd.Series([
            bb['price'].iloc[0],
            bb['size'].iloc[0],
            ba['price'].iloc[0],
            ba['size'].iloc[0]
        ])

    def _log_it(self, message:str, log_type:str="INFO", print_comb:bool=False):
        """
        Main logging method.

        Args:
            - message : The text to log
            - log_type : The logging level (INFO, WARNING, ERROR)
            - print_comb : Displays the markets pairs combination in the message if True (e.g. M1_PAIR1_M2_PAIR2)
                           `False` as default value
        """
        levels = {
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
        }
        log_func = levels.get(log_type.upper(), self.logger.debug)
        log_func(f"{self.comb} - " + message if print_comb else message)

    def _std_basepath(self):
        """
        Standardize output path to avoid undesidred output paths
        """
        if self.OUTPUT_BASEPATH[-1] == "/":
            return self.OUTPUT_BASEPATH
        return self.OUTPUT_BASEPATH + "/"
    
    def _extract_model_params(self, model:pd.DataFrame):

        def _set_gpd_params(gpd_dist):
            """
            Returns the GPD distribution parameters and 
            a flag if the extreme value theory should be applied or not

            Args:
                -gpd_dist: Fitted GPD distribution from SciPy or NaN
            """
            if pd.notna(gpd_dist):
                return gpd_dist.args[0], *gpd_dist.kwds.values(), True
            else:
                return [np.nan for i in range(3)] + [False]

        mdl = model.iloc[0]

        #General params
        self.m1_venue, self.pair_m1 = mdl['market_1'], mdl['pair_1']
        self.m2_venue, self.pair_m2 = mdl['market_2'], mdl['pair_2']

        #Autoregressive lags
        self.ar_lags = mdl[self.AR_LAGS_FIELD]

        #Optional Extreme Value Theory
        gpd_right = mdl['gpd_dist_right']
        gpd_left = mdl['gpd_dist_left']
        #Set the GPD parameters
        self.z_tail_left, self.z_tail_right  = (mdl['u_threshold_left'], mdl['u_threshold_right'])
        self.gpd_r_shape, self.gpd_r_loc, self.gpd_r_scale, self.extremes_right = _set_gpd_params(gpd_right)
        self.gpd_l_shape, self.gpd_l_loc, self.gpd_l_scale, self.extremes_left = _set_gpd_params(gpd_left)

        #Mandatory Students-t distribution
        #Set the distribution parameters
        self.d_f, self.loc, self.scale = mdl['dist']

        #Allowed distribution side to trade
        self.trade_left = mdl['left_trade']
        self.trade_right = mdl['right_trade']
        

    def _load_orderbook(self, model:pd.DataFrame, print_comb:bool=False):
        """
        Load orderbooks for a single pairs/markets combination.

        Args:
            - model : a unitary slice of the `self.models` DataFrame
        """


        cols = [
                "m_p",
                "m1_sourcepath_signal",
                "m2_sourcepath_signal",
                "m1_sourcepath_execution",
                "m2_sourcepath_execution",
                ]

        model = model[cols].iloc[0]

        self._log_it(f"Importing signal and execution orderbooks", print_comb=print_comb)

        #Order books for trading signals (typically before the actual execution time)
        self.m1_bba, self.m2_bba = [pd.read_pickle(path).sort_values(self.FIELD_TIMESTAMP) for path in model[cols[1:3]]]

        #Order books for trade execution (typically after the signal time)
        self.order_book_m1, self.order_book_m2 = [pd.read_pickle(path).sort_values(self.FIELD_TIMESTAMP) for path in model[cols[3:5]]]

        self._log_it(f"Orderbooks imported", print_comb=print_comb)

    def _trunc_orderbooks(self, t_delta:float=None):
        """
        Truncate the orderbook datasets so they all start and finish at the same timestamp

        Args:
            - t_delta : allows for a slight time deviation to avoid unnecessary time jumps ; the risk of lookahead bias is mitigated by `_manage_lookahead`
        """

        if not t_delta:
            t_delta = self.TRUNC_TIME_DELTA

        #Take from the first timestamp of the latest dataset start
        self.m1_bba =self.m1_bba[
            (self.m1_bba[self.FIELD_TIMESTAMP] >= self.m2_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) & 
            (self.m1_bba[self.FIELD_TIMESTAMP] >= self.order_book_m1[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.m1_bba[self.FIELD_TIMESTAMP] >= self.order_book_m2[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta))
            ].reset_index(drop=True)
        self.m2_bba = self.m2_bba[
            (self.m2_bba[self.FIELD_TIMESTAMP] >= self.m1_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.m2_bba[self.FIELD_TIMESTAMP] >= self.order_book_m1[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.m2_bba[self.FIELD_TIMESTAMP] >= self.order_book_m2[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta))
            ].reset_index(drop=True)
        self.order_book_m1 = self.order_book_m1[
            (self.order_book_m1[self.FIELD_TIMESTAMP] >= self.order_book_m2[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.order_book_m1[self.FIELD_TIMESTAMP] >= self.m1_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.order_book_m1[self.FIELD_TIMESTAMP] >= self.m2_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta))
        ].reset_index(drop=True)
        self.order_book_m2 = self.order_book_m2[
            (self.order_book_m2[self.FIELD_TIMESTAMP] >= self.order_book_m1[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.order_book_m2[self.FIELD_TIMESTAMP] >= self.m1_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta)) &
            (self.order_book_m2[self.FIELD_TIMESTAMP] >= self.m2_bba[self.FIELD_TIMESTAMP].min() - pd.to_timedelta(t_delta))
        ].reset_index(drop=True)

        #Take until the latest timestamp of the shortest dataset
        self.m1_bba = self.m1_bba[
            (self.m1_bba[self.FIELD_TIMESTAMP] <= self.m2_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) & 
            (self.m1_bba[self.FIELD_TIMESTAMP] <= self.order_book_m1[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.m1_bba[self.FIELD_TIMESTAMP] <= self.order_book_m2[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta))
            ].reset_index(drop=True)
        self.m2_bba = self.m2_bba[
            (self.m2_bba[self.FIELD_TIMESTAMP] <= self.m1_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.m2_bba[self.FIELD_TIMESTAMP] <= self.order_book_m1[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.m2_bba[self.FIELD_TIMESTAMP] <= self.order_book_m2[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta))
            ].reset_index(drop=True)
        self.order_book_m1 = self.order_book_m1[
            (self.order_book_m1[self.FIELD_TIMESTAMP] <= self.order_book_m2[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.order_book_m1[self.FIELD_TIMESTAMP] <= self.m1_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.order_book_m1[self.FIELD_TIMESTAMP] <= self.m2_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta))
        ].reset_index(drop=True)
        self.order_book_m2 = self.order_book_m2[
            (self.order_book_m2[self.FIELD_TIMESTAMP] <= self.order_book_m1[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.order_book_m2[self.FIELD_TIMESTAMP] <= self.m1_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta)) &
            (self.order_book_m2[self.FIELD_TIMESTAMP] <= self.m2_bba[self.FIELD_TIMESTAMP].max() + pd.to_timedelta(t_delta))
        ].reset_index(drop=True)

    def _reshape_orderbooks(self):
        """
        Reshape the M1 and M2 orderbooks in a time-aware manner so they all have the same size as `m1_bba`.
        When time gaps exist, the record on the closest timestamp is duplicated
        """

        #Get the closest timestamp on each record with the associated index
        time_diffs = find_closest_across_dataframes(
            self.m1_bba[[self.FIELD_TIMESTAMP]], 
            {
                'm2_bba':self.m2_bba[[self.FIELD_TIMESTAMP]],
                'ob_m1' : self.order_book_m1[[self.FIELD_TIMESTAMP]],
                'ob_m2' : self.order_book_m2[[self.FIELD_TIMESTAMP]]
            }
            )

        #Time-aware reshape
        self.m2_bba = self.m2_bba.loc[time_diffs['m2_bba_index']].reset_index(drop=True)
        self.order_book_m1 = self.order_book_m1.loc[time_diffs['ob_m1_index']].reset_index(drop=True)
        self.order_book_m2 = self.order_book_m2.loc[time_diffs['ob_m2_index']].reset_index(drop=True)


    def _manage_lookahead(self):
        """
        Mitigate the lookahead bias by excluding data points stamped after M1 timestamp.
        `max_lookahead` acts as a tolerance to limit the lookahead bias ; it defines the maximum allowed time difference in datasets granularity time unit
        """
        max_lookahead = self._second_to_precision(self.LOOKAHEAD_TOLERANCE, self.TIMESTAMP_PRECISION)
        mask_bba = ((pd.to_numeric(self.m1_bba['time_exchange']) - pd.to_numeric(self.m2_bba['time_exchange'])) < (-max_lookahead))
        mask_ob =((pd.to_numeric(self.order_book_m1['time_exchange']) - pd.to_numeric(self.order_book_m2['time_exchange'])) < (-max_lookahead))

        self.m1_bba, self.m2_bba, self.order_book_m1, self.order_book_m2 = self.m1_bba.loc[~(mask_bba | mask_ob)].reset_index(drop=True), self.m2_bba.loc[~(mask_bba | mask_ob)].reset_index(drop=True), \
            self.order_book_m1.loc[~(mask_bba | mask_ob)], self.order_book_m2.loc[~(mask_bba | mask_ob)].reset_index(drop=True)

    def _calc_px_delta(self):
        """
        Calculates the price deltas between M1 and M2 in bps of M1.
        The results are store in `m1_m2` and are used for trade signals
        """

        self.m1_bba[['bb_price', 'bb_size', 'ba_price', 'ba_size']] = self.m1_bba.apply(self._bba_from_l2, axis=1)
        self.m2_bba[['bb_price', 'bb_size', 'ba_price', 'ba_size']] = self.m2_bba.apply(self._bba_from_l2, axis=1)

        #Compute mid quotes
        self.m1_bba['mid'], self.m2_bba['mid'] = (self.m1_bba['bb_price'] + self.m1_bba['ba_price'])/2,(self.m2_bba['bb_price'] + self.m2_bba['ba_price'])/2 

        #Compute price detlas
        cols = ['symbol_id', 'time_exchange', 'time_client', 'mid']
        m1 = self.m1_bba[cols]
        m1.columns = [c + "_m1" for c in cols]
        m2 = self.m2_bba[cols]
        m2.columns = [c + "_m2" for c in cols]
        self.m1_m2 = pd.concat([m1, m2], axis=1)
        self.m1_m2['px_delta'] = ((self.m1_m2['mid_m1'] - self.m1_m2['mid_m2'])/self.m1_m2['mid_m1']*1e4)

        #Mandatory check
        self._validate_sample_size()

    def _predict_stats(self, ar_model:univariate.base.ARCHModelResult):
        """
        Predict means and volatilities using the AR Model parameters

        Args:
            - ar_model : The AutoRegressive model generated by arch
        """

        #Instatiate an ARCH model based on the model parameters for inference
        ar_model = arch_model(self.m1_m2['px_delta'], mean=self.AR_MODEL_MEAN_TYPE, vol=self.AR_MODEL_TYPE,
                            p=1, q=1, lags=self.ar_lags, dist=self.MAIN_DISTRIBUTION).fix(ar_model.params)

        #Predict next immediate means and volatilities
        last_lag = max(self.ar_lags)
        self.means = ar_model.forecast(horizon=1, start=last_lag, method='simulation').mean.to_numpy().flatten()
        self.vols = (ar_model.forecast(horizon=1, start=last_lag, method='simulation').variance**(1/2)).to_numpy().flatten()

    def _ar_lags_trim(self):
        """
        Trims the datasets to start after the last autregressive lag of the model so it matches `self.means` and `self.vols` sizes.     
        """

        #Last autregressive lag
        start_index = max(self.ar_lags)

        #Trim datasets
        self.order_book_m1 = self.order_book_m1[start_index:].reset_index(drop=True)
        self.order_book_m2 = self.order_book_m2[start_index:].reset_index(drop=True)
        self.m1_m2 = self.m1_m2.loc[start_index:].reset_index(drop=True)

    def _convert_exec_dset(self):
        """
        Convert the L2 orderbooks for trade execution in a structure and format compatible with the hftbacktest input data specs.
        """

        str_start_dtime = self.order_book_m1.iloc[0]['time_client'].strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

        self.order_data_m1, _ = l2_convert([self.order_book_m1], fix_interval_nanos=300*1e9, clear_offset=-60*1e9, start_dtime=str_start_dtime)
        self.order_data_m2, _ = l2_convert([self.order_book_m2], fix_interval_nanos=300*1e9, clear_offset=-60*1e9, start_dtime=str_start_dtime)

    def _assign_fees(self):
        """
        Assign the fees in bps to the M1 and M2 markets
        """

        def handle_fee_values(fee, venue:str):
            if fee is None:
                self._log_it(f"{venue} has no fee",log_type="WARNING")
                return {
                    "maker" : 0.0,
                    "taker" : 0.0
                }
            elif "maker" not in fee.keys():
                self._log_it(f"{venue} has no maker fee",log_type="WARNING")
                fee['maker'] = 0.0
            elif "taker" not in fee.keys():
                self._log_it(f"{venue} has no taker fee",log_type="WARNING")
                fee['taker'] = 0.0
            return fee


        mkt_type1 = "futures" if self.pair_m1[-2:] == ".P" else "spot"
        mkt_type2 = "futures" if self.pair_m2[-2:] == ".P" else "spot"

        self.fees_m1 = handle_fee_values(get_fees(venue=self.m1_venue, market_type=mkt_type1, mkt_fees_file=self.MARKET_FEES_JSON_PATH), self.m1_venue)
        self.fees_m2 = handle_fee_values(get_fees(venue=self.m2_venue, market_type=mkt_type2, mkt_fees_file=self.MARKET_FEES_JSON_PATH), self.m2_venue)


    def _validate_sample_size(self):
        """
        Check the minimum required sampled size is reached. A too narrow sample size undermines the reliability of the backtest results
        This has to be called even if MIN_SAMPLE_SIZE is 0
        """

        self.sample_size = self.m1_m2.shape[0]
        if self.sample_size < self.MIN_SAMPLE_SIZE:
            raise ValueError(f"{self.comb} - Too few samples. Only {self.sample_size} remaining while the minumum is {self.MIN_SAMPLE_SIZE}")

    def _validate_clean_input_data(self, print_comb:bool=False):
        """
        Validate the input datasets have the expected size. This is crutial for aligment between signal and trade datasets during the backtesting operations.
        Validate the events in the `order_data` are in the expected sequence.

        Args:
            - print_comb : Displays the markets pairs combination in logs
                           Default `False`
        """

        self._log_it("Validate input data shapes...", print_comb=print_comb)

        shape_chk = {
        "m1" : (self.m1_m2.shape[0] == self.order_book_m1.shape[0]),
        "m2" : (self.m1_m2.shape[0] == self.order_book_m2.shape[0]),
        "means" : (self.means.shape[0] == self.m1_m2.shape[0]),
        "vols" : (self.vols.shape[0] == self.m1_m2.shape[0]),
        }
        for k,v in shape_chk.items():
            if not v:
                raise ValueError(f"The size of datasets don't match for {k}")

        self._log_it("Validate input data sequence...", print_comb=print_comb)
        #Use the hftbacktest native function to validate the input data sequence
        validate_event_order(self.order_data_m1)
        validate_event_order(self.order_data_m2)

        self._log_it("All validations passed", print_comb=print_comb)
    
    def _set_hftbt_vars(self):
        """
        Declare and set the parameters for the backtester and its recorder
        """

        asset_m1 = asset_const(
            dataset=self.order_data_m1[1:], #Start after the first clear event
            exchange=self.m1_venue,
            instrument=self.pair_m1,
            fee_model='trading_value',
            fee_maker=self.fees_m1['maker']/100,
            fee_taker=self.fees_m1['taker']/100,
            last_trades_capacity=0,
            )
        asset_m2 = asset_const(
            dataset=self.order_data_m2[1:], #Start after the first clear event
            exchange=self.m2_venue,
            instrument=self.pair_m2,
            fee_model='trading_value',
            fee_maker=self.fees_m2['maker']/100,
            fee_taker=self.fees_m2['taker']/100,
            last_trades_capacity=0,
        )

        #Backtester object
        self.hbt = HashMapMarketDepthBacktest([asset_m1, asset_m2])
        #Associated recorder object
        self.recorder = Recorder(2, 100_000)

    def _main_loop(self, i, q):
        """
        Runs for a single markets pairs combination. It is expected to be the target function of multiprocessing.
        This method performs the following steps:
            1. Loads statistical parameters
            2. Prepares and validate all the datasets
            3. Instantiate the required objects for Hftbacktest
            4. Run the backtest through the core backtesting object
        
        Args:
            - i: The i-th markets pairs combination to run (zeroed lower bound)
            - q: The multiprocessing Queue to communicate the results to the main instance
        """
        from src.backtest_cores import CoreLstmr #Avoids circular references

        self._log_it(f"Loading model {i}...")
        self.model = self.models.iloc[[i],:]
        self.comb = self.model.iloc[0]['m_p']
        self._log_it(f"Loaded on {multiprocessing.Process().name}", print_comb=True)

        self._log_it(f"Fetching {self.comb} parameters...")
        self._extract_model_params(self.model)

        self._load_orderbook(self.model, print_comb=True)

        self._log_it("Cleaning and sorting orderbook datasets...", print_comb=True)
        self._trunc_orderbooks()

        self._reshape_orderbooks()
        self._manage_lookahead()
        self._log_it(f"orderbooks records sizes - M1 Signal {self.m1_bba.shape[0]} | M1 Execution {self.order_book_m1.shape[0]} | M2 Signal {self.m2_bba.shape[0]} | M2 Execution {self.order_book_m2.shape[0]}", print_comb=True)


        self._log_it("Calculating signal price deltas...", print_comb=True)
        self._calc_px_delta()

        self._log_it("Predicting immediate means and volatilities...", print_comb=True)
        self._predict_stats(self.model.iloc[0]['model'])
        self._log_it("Done predicting", print_comb=True)

        self._log_it("Preparing L2 orderbook data for Hftbacktest...", print_comb=True)
        self._ar_lags_trim()
        self._convert_exec_dset()
        self._log_it("Done", print_comb=True)

        self._log_it("Final datasets validations", print_comb=True)
        self._validate_clean_input_data(print_comb=True)

        self._log_it(f"Setting markets fees using source {self.MARKET_FEES_JSON_PATH}", print_comb=True)
        self._assign_fees()
        self._log_it(f"M1 : {self.fees_m1} | M2 : {self.fees_m2}", print_comb=True)

        
        self._log_it("Spawning a new backtester instance...", print_comb=True)
        btest = CoreLstmr(**self.__dict__)
        self._log_it("Running backtest...", print_comb=True)
        btest_res = btest._backtest(
            px_delta = self.m1_m2['px_delta'],
            mid_quote = self.m1_m2[['mid_m1', 'mid_m2']]
        )

        #Capture recorder results in main process
        q.put(btest_res)

        self._log_it(f"Finished {self.comb}")

    #Parameters setters methods
    def set_field_timestamp(self, value:str):
        """
        Field name to be used as reference time in L2 orderbook datasets
        """
        self.FIELD_TIMESTAMP = value

    def set_trunc_time_delta(self, value:str):
        """
        Time deviation to avoid unnecessary time jumps while leveling L2 orderbook datasets
        Value is expressed as time in timeunit (e.g. for 10 milliseconds the value should be "10ms")
        """
        self.TRUNC_TIME_DELTA = value

    def set_lookahead_tolerance(self, value:float):
        """
        Time tolerance limit applied when mitigating lookahead bias.
        Lookahead is mitigated by enforcing M2 time <= M1 time across L2 orderbooks. 
        This may exclude too much data on some datasets ; low tolerance helps getting a wide enough sample size 
        while limiting the impact of lookahead.
        Value should be expressed in seconds (decimal values allowed)
        """
        self.LOOKAHEAD_TOLERANCE = value

    def set_timestamp_precision(self, value:str):
        """
        Timestamp granularity used for the strategy
        Value should be a time unit (e.g. "ns" for nanoseconds)
        """
        self.TIMESTAMP_PRECISION = value

    def set_main_distribution(self, value:str):
        """
        Type of the mandatory distribution for trade signals.
        Only Student-t is currently supported (value: "t")
        """
        self.MAIN_DISTRIBUTION = value

    def set_ar_model_mean_type(self, value:str):
        """
        Mean type used when fitting the Time Series model
        Only AutoRegressive mean is currently supported (value : "AR")
        """
        self.AR_MODEL_MEAN_TYPE = value

    def set_ar_model_type(self, value:str):
        """
        Type of AutoRegressive model used  when fitting the Time Series model
        Suported values : "FIGARH", "GARCH"
        """
        self.AR_MODEL_TYPE = value
    
    def set_ar_lags_field(self, value:str):
        """
        Field name that contains the list of AutoRegressive lags in the AR model.
        """
        self.AR_LAGS_FIELD = value

    def set_min_sample_size(self, value:int):
        """
        Minimum required sample size on the L2 orderbook datasets to run the backtest.
        """
        self.MIN_SAMPLE_SIZE = value

    def set_market_fees_json_path(self, value:str):
        """
        Path to the JSON file containing the Maker and Taker fees (in bps) for each market.
        """
        self.MARKET_FEES_JSON_PATH = value

    def set_equity(self, value:np.int64):
        """
        Value of avialable equity on market expressed in pairs quote currency.
        Note the total equity used for the strategy is `EQUITY` * 2 (as the same `EQUITY` is applied to M1 and M2)
        """
        self.EQUITY = value

    def set_max_eq_loss(self, value:float):
        """
        The value of the maximum acceptable loss on a single trade.
        """
        self.MAX_EQ_LOSS = value

    def set_max_loss_prob(self, value:float):
        """
        The maximum allowed predicted probability of experiencing at least `MAX_EQ_LOSS`
        Value should be expressed as decimal.
        """
        self.MAX_LOSS_PROB = value

    def set_step_time_inteval(self, value:float):
        """
        The time interval between each set of decisions can be taken while running the backtest.
        Value should be expressed in seconds (decimals allowed)
        """
        self.STEP_TIME_INTEVAL = value

    def set_min_probability_entry_signal(self, value:float):
        """
        The minimum required predicted probability of breakeven to enter a new position.
        Value should be expressed as decimal
        """
        self.MIN_PROBABILITY_ENTRY_SIGNAL = value

    def set_max_probability_stop_loss_signal(self, value:float):
        """
        The maximum allowed predicted probability of breakeven that triggers a stop loss
        Value should be expressed as decimal
        """
        self.MAX_PROBABILITY_STOP_LOSS_SIGNAL = value

    def set_output_basepath(self, value:str):
        """
        The path of the directory to save all the outputs.
        """
        self.OUTPUT_BASEPATH = value

    def show_params(self):
        "Pretty print all the parameters"

        #Get all class arguments and filter on params
        args = self.__dict__
        params = {k:v for k,v in args.items() if k.upper() == k}

        key_header, val_header = "Parameter", "Value"
        key_width = max(len(str(k)) for k in list(params.keys()) + [key_header])
        val_width = max(len(str(v)) for v in list(params.values()) + [val_header])

        table_width = key_width + val_width + 7  # borders/padding

        # Title bar
        print("┌" + "─" * (table_width - 2) + "┐")
        print("│" + "STRATEGY PARAMETERS".center(table_width - 2) + "│")
        print("├" + "─" * (key_width + 2) + "┬" + "─" * (val_width + 2) + "┤")

        # Header row
        print(f"│ {key_header.center(key_width - 2):<{key_width}} │ {val_header.center(val_width - 2):<{val_width}} │")
        print("├" + "─" * (key_width + 2) + "┼" + "─" * (val_width + 2) + "┤")

        # Data rows
        for k, v in params.items():
            print(f"│ {str(k):<{key_width}} │ {str(v):<{val_width}} │")

        print("└" + "─" * (key_width + 2) + "┴" + "─" * (val_width + 2) + "┘")

    def run(self, produce_stats:bool=True, num_workers:int=8):
        """
        Run the strategy accross models

        Args:
            - num_workers : The number of workers (processes) to be run in parallel. 
                            Multi-processed if > 1; single processed otherwise
                            Default value : 8
            - produce_stats : Generate the result statistics (default True)
        """

        #Ensure the input exists
        if "models" not in self.__dict__.keys():
            raise RuntimeError("Input models not found ! Did you run `import_models` ? ")

        #Validate method's arguments
        if num_workers < 1 or not isinstance(num_workers, int):
            raise ValueError(f"Invalid num_workers:{num_workers}. Expected integer value >= 1")

        #Hftbacktest recorder results
        self.res = {}

        #Prepare batches
        batches = []
        procs_idx = [i for i in range(self.models_count)]
        for i in range(ceil(self.models_count / num_workers)):
            batches.append(procs_idx[i * num_workers: (i + 1) * num_workers])


        procs = []
        q = multiprocessing.Queue()

        self._log_it("Starting backtesting...")

        #Run in batches where each batch <= `num_workers`
        for b in batches:
            batch_procs = []   
            for i in b:
                p = multiprocessing.Process(
                    target=self._main_loop,
                    name=f"Process {i}",
                    daemon=True,
                    args=(i,q)
                )
                p.start()
                procs.append(p)
                batch_procs.append(p)

            #Fetch returned values
            for _ in batch_procs:
                self.res.update(q.get())

        for p in procs:
            p.join()

        self._log_it("Successfully completed the backtesting operations !")
        
        #Generate statistics if instructed
        if produce_stats:
            self._log_it("Generating results statistics...")
            self.stats = StatsLstmr(
                strategy_results=self.res,
                equity=self.EQUITY,
                num_workers=num_workers
            )
            self.stats.produce_stats(base_directory=self.OUTPUT_BASEPATH)


    def import_models(self, model_pckl_path:str):
        """
        Import the models to run the backtesting on.

        Args:
            - model_pckl_path : The path of the Pandas DataFrame model saved in pickle format
        """

        columns_val = ['vol_model',
                        'p',
                        'q',
                        'mean_type',
                        'model',
                        'm_p',
                        'convergence_flag',
                        'AIC',
                        'BIC',
                        'market_1',
                        'pair_1',
                        'market_2',
                        'pair_2',
                        self.AR_LAGS_FIELD,
                        'lag_h',
                        'pval',
                        'u_threshold_left',
                        'gpd_dist_left',
                        'u_threshold_right',
                        'gpd_dist_right',
                        'dist_name',
                        'KS p-value',
                        'dist',
                        "m1_sourcepath_signal",
                        "m1_sourcepath_execution",
                        "m2_sourcepath_signal",
                        "m2_sourcepath_execution",
                        "left_trade",
                        "right_trade",]

        self.models = pd.read_pickle(model_pckl_path)

        self.models = self.models[columns_val]

        self.models_count = self.models.shape[0]

        print(f"Imported {self.models_count} models")