
import numpy as np
import pandas as pd
from hftbacktest import GTC, MARKET
from scipy import stats

from src.backtest_pipelines import LiquidStackingMeanReversion

class CoreLstmr(LiquidStackingMeanReversion):
    """
    Core class for the Liquid Stacking Mean Reversion strategy.
    Class inherits from all arguments and method from the instatiated parent and is designed to be run as multiprocessed with segregated arguments
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_hftbt_vars()

    def _print_orders(self, order_cat:str, order_id:np.int64, px_init_m1:list=[], px_init_m2:list=[]):
            """
            Print orders when it hit the markets

            -Args:
                - order_cat : Order category (e.g. Entry, Stop loss, Take profit)
                - order_id : Current unique order id
                - px_init_m1: List of prices on M1 at entries
                - px_init_m2: List of prices on M2 at entries
            """
            orders_m1, orders_m2 = self.hbt.orders(0).get(order_id), self.hbt.orders(1).get(order_id)
            print(f"""
            combination : {self.comb} ; order_id : {order_id} ; Order category : {order_cat} ; Market : M1 ; Exchange timestamp {pd.to_datetime(orders_m1.exch_timestamp)} ; \
            Executed price : {orders_m1.exec_price} ; Executed quantity : {orders_m1.exec_qty} ; Side : {orders_m1.side} ; Status {orders_m1.status} ; Init M1 Price {px_init_m1}
            """)
            print(f"""
            combination : {self.comb} ; order_id : {order_id} : Order category : {order_cat} ; Market : M2 ; Exchange timestamp {pd.to_datetime(orders_m2.exch_timestamp)} ; \
            Executed price : {orders_m2.exec_price} ; Executed quantity : {orders_m2.exec_qty} ; Side : {orders_m2.side} ; Status {orders_m2.status} : Init M1 Price {px_init_m2}
            """)

    def _update_pred_stats(self):
        """
        Update statistics for execution decisions: 
            - price delta target : the take profits target (i.e. the expected value of the price delta given breakeven)
            - fee recover probability : the probability of break even on time t+1
        """

        next_i = min((self.i + 1), (self.vols.shape[0]-1))
        vol = self.vols[next_i]
        mean = self.means[next_i]

        #In case there is no open positon at t=i
        if self.out_trades.empty:
            return

        px_delta = self.out_trades['px_delta'].to_numpy()
        fee = self.out_trades['fee'].to_numpy()
        sign = self.out_trades['side'].to_numpy()

        neg_mask = sign < 0
        pos_mask = sign > 0

        #If any long exposure
        if pos_mask.any():
            z = (px_delta[pos_mask] + fee[pos_mask] - mean) / vol #Z-Value at breakeven
            if self.extremes_right:
                pos_mask_ext = z > self.z_tail_right
                pos_mask = z <= self.z_tail_right
                #Only when in the right-tail extreme values territory
                if pos_mask_ext.any():
                    z = (px_delta[pos_mask_ext] + fee[pos_mask_ext] - mean) / vol #Z-Value at breakeven for trades in the right-tail extreme values territory
                    abs_z = abs(z)                 
                    fee_recover_prob = stats.genpareto.sf(abs_z, self.gpd_r_shape, loc=self.gpd_r_loc, scale=self.gpd_r_scale) * stats.t.sf(self.z_tail_right, self.d_f) #≈(1-CDF(Z~GPD(Shape, volatility, mean)))
                    cond_mean_std = (self.gpd_r_scale + abs_z) / (1 - self.gpd_r_shape) #Expected Z-Value given breakeven for GPD

                    self.out_trades.loc[pos_mask_ext, 'delta_target'] = mean + vol * cond_mean_std #Scale to get the expected value of the price delta
                    self.out_trades.loc[pos_mask_ext, 'fee_recover_prob'] = fee_recover_prob

            z = (px_delta[pos_mask] + fee[pos_mask] - mean) / vol
            sf = stats.t.sf(z, self.d_f) #~(1-CDF(Z~T(degrees_of_freedom, 0, 1)))
            pdf = stats.t.pdf(z, self.d_f)
            cond_mean_std = (self.d_f + z**2) / (self.d_f - 1) * pdf / sf #Expected Z-Value given breakeven for Student-T distribution (i.e. E[X | X > u])
            self.out_trades.loc[pos_mask, 'delta_target'] = mean + vol * cond_mean_std #Scale to get the expected value of the price delta
            self.out_trades.loc[pos_mask, 'fee_recover_prob'] = stats.t.cdf(z, self.d_f) 


        #If any short exposure
        if neg_mask.any():
            z = (px_delta[neg_mask] - fee[neg_mask] - mean) / vol
            if self.extremes_left:
                neg_mask_ext = z < -self.z_tail_left
                neg_mask = z >= -self.z_tail_left
                #Only when in the left-tail extreme values territory
                if neg_mask_ext.any():
                    z = (px_delta[neg_mask_ext] - fee[neg_mask_ext] - mean) / vol #Z-Value at breakeven for trades in the left-tail extreme values territory
                    abs_z = abs(z) 
                    fee_recover_prob = stats.genpareto.sf(abs_z, self.gpd_l_shape, loc=self.gpd_l_loc, scale=self.gpd_l_scale) * stats.t.sf(self.z_tail_left, self.d_f) #≈(1-CDF(Z~GPD(Shape, volatility, mean)))
                    cond_mean_std  = -((self.gpd_l_scale + abs_z) / (1 - self.gpd_l_shape)) #Expected Z-Value given breakeven for GPD

                    self.out_trades.loc[neg_mask_ext, 'delta_target'] = mean + vol * cond_mean_std #Scale to get the expected value of the price delta
                    self.out_trades.loc[neg_mask_ext, 'fee_recover_prob'] = fee_recover_prob

            z = (px_delta[neg_mask] - fee[neg_mask] - mean) / vol
            cdf = stats.t.cdf(z, self.d_f)
            pdf = stats.t.pdf(z, self.d_f)
            cond_mean_std = -(self.d_f + z**2) / (self.d_f - 1) * pdf / cdf #Expected Z-Value given breakeven for Student-T distribution (i.e. E[X | X < u])
        

            self.out_trades.loc[neg_mask, 'delta_target'] = mean + vol * cond_mean_std #Scale to get the expected value of the price delta
            self.out_trades.loc[neg_mask, 'fee_recover_prob'] = cdf


    def _entry_exec(self, d, mid_quotes:tuple, last_spread_m1:float, last_spread_m2:float):
        """
        Fire orders to get additional exposures when the signal detects potentially profitable market conditions

        Args:
            - d : Price delta at signal time
            - mid_quotes: M1 and M2 mid prices at signal time
            - last_spread_m1 : Observed spread on M1 at t-1
            - last_spread_m2 : Observed spread on M2 at t-1
        """
        
        #Mid-quote at signal time
        mid_m1, mid_m2 = mid_quotes
        
        fee = (mid_m1 * (self.fees_m1['taker']/100 + self.fees_m1['maker']/100) + mid_m2 * (self.fees_m2['taker']/100 + self.fees_m2['maker']/100))/mid_m1*1e4 #Expected fees in bps of M1 at signal time
        i_next = min((self.i + 1), (self.sample_size - 1))
        spread_impact = ((last_spread_m1 / mid_m1) + (last_spread_m2 / mid_m1)) * 1e4 #Spread impact is expected to be the same as the observered spread at t-1
        order_id = self.sample_size * 2 + self.i

        #BUY M1 - SELL M2
        #Buy when the signal price delta adjusted for spread and fees is below the predicted autoregressive mean and no short exposure is outstanding
        if (((d + spread_impact) < self.means[self.i]) and self.out_trades.empty) or ((d + spread_impact) < self.means[self.i] and (self.out_trades['qty'] * self.out_trades['side']).sum() > 0):

            if self.trade_left:  #Is it allowed to trade the left side of the distribution as per model's specs ?

                beven_z_score = ((d + fee) - self.means[i_next]) / self.vols[i_next] #Breakeven Z-Value at predicted t+1

                if self.extremes_right and (beven_z_score > self.z_tail_right):
                    """
                    Under current extreme values the breakeven probability is the joint probability of being above the right extreme value on a Student-t distribution and the probability of being above the t+1 breakeven Z-Value on a GPD.
                    So fee_beven_prob = P(t_ext ∩ g_z)
                    With :
                        -t_ext ≈ 1-CDF(extreme_threshold~T(0, 1))
                        -g_z ≈ 1-CDF(breakeven_z_score~GPD(degrees_of_freedom, 0, 1))
                    """
                    fee_beven_prob = stats.t.sf(self.z_tail_right, df=self.d_f) * stats.genpareto.sf(beven_z_score, self.gpd_r_shape, self.gpd_r_loc, self.gpd_r_scale)
                else:
                    #Otherwise fee_beven_prob ≈ 1-CDF(breakeven_z_score~T(degrees_of_freedom, mean, volatility))
                    fee_beven_prob = stats.t.sf(d + fee, loc=self.means[i_next], scale=self.vols[i_next], df=self.d_f)

                #Trade signal : Place order when the breakeven probability is above the required minimum
                if fee_beven_prob > self.MIN_PROBABILITY_ENTRY_SIGNAL:

                    #SIZING
                    #Calculate trade size according to volume constraints and risk apetite
                    #Leverage should be set to have MAX_LOSS_PROB probability of experiencing a loss of at least MAX_EQ_LOSS

                    #Compute the minimum losss in price delta at the MAX_LOSS probability

                    #Only when extreme value theory applies to the left tail of the distribution
                    if self.extremes_left:
                        thr_prob = stats.t.cdf((-self.z_tail_left), loc=0, scale=1, df=self.d_f) #Probability of ending in extreme values in the Student-t distribution
                        """
                        Under extreme values theory the maximum acceptable loss occurs at a price delta corresponding to the conditional probability of experiencing MAX_LOSS_PROB given the probability of being in the extreme left tail (proven that Z_max_loss < -extreme_tail)
                        Therefore : `max_px_delta_loss` = 1/CDF:(quantile) ~ GPD(shape, gpd_mean, gpd_vol) * volatility - mean
                        With :
                            - quantile = 1 - P(MAX_LOSS | P(extreme_tail)) = 1 - [MAX_LOSS / CDF(-extreme_tail) ~ T(degrees_of_freedom, 0, 1)]
                        """
                        max_px_delta_loss = (stats.genpareto.ppf(1 - (self.MAX_LOSS_PROB / thr_prob), self.gpd_l_shape ,loc=self.gpd_l_loc, scale=self.gpd_l_scale) * self.vols[i_next]) - abs(self.means[i_next])
                    #Otherwise
                    else:
                        #`max_px_delta_loss` = 1/CDF:(quantile) ~ T(degrees_of_freeddom, mean, volatility)
                        max_px_delta_loss = stats.t.ppf(self.MAX_LOSS_PROB, loc=self.means[i_next], scale=self.vols[i_next], df=self.d_f) 

                    max_px_deviation = max_px_delta_loss * mid_m1 / 1e4 #Maximum acceptable deviation in bps
                    #Maximum leverage is the leverage at which MAX_EQ_LOSS of EQUITY is lost given the maximum acceptable deviation in bps
                    #As equity_loss = (loss_ratio * EQUITY), the maximum size is equity_loss / (p_t - p_t-1)
                    size = (-self.MAX_EQ_LOSS * self.EQUITY) / max_px_deviation

                    qty = min(self.depth_m1.best_ask_qty, abs(size))

                    #Fire orders
                    self.hbt.submit_buy_order(0, order_id, self.depth_m1.best_ask, qty, GTC, MARKET, True) #M1
                    self.hbt.submit_sell_order(1, order_id, self.depth_m2.best_bid, qty, GTC, MARKET, True) #M2

                    exec_px_m1 = self.hbt.orders(0).get(order_id).exec_price
                    exec_px_m2 = self.hbt.orders(1).get(order_id).exec_price
                    exec_delta = ((exec_px_m1 - exec_px_m2)/exec_px_m1*1e4) #Observed price delta at execution

                    #Hydrate outstanding trades
                    new_entry = pd.Series(
                            [
                            order_id,
                            exec_px_m1,
                            exec_px_m2,
                            exec_delta,
                            abs(self.hbt.orders(0).get(order_id).qty),
                            1.0,
                            fee,
                            self.means[self.i],
                            np.nan,
                            np.nan
                            ],
                            index=self.out_cols
                        )
                    self.out_trades = pd.concat([self.out_trades, pd.DataFrame(new_entry).T.set_index('order_id')], axis=0)

                    self._print_orders(order_cat="Entry", order_id=order_id)

                    #Hydrate slippage
                    s = pd.Series(
                        [
                            mid_m1,
                            self.depth_m1.best_ask,
                            mid_m2,
                            self.depth_m2.best_bid,
                            d*mid_m1/1e4,
                            (self.depth_m1.best_ask - self.depth_m2.best_bid),
                            ((self.depth_m1.best_ask) - ((self.depth_m1.best_bid + self.depth_m1.best_ask)/2)),
                            (((self.depth_m2.best_bid + self.depth_m2.best_ask)/2) - (self.depth_m2.best_bid)),
                        ],
                        index=self.sl_cols
                    )
                    self.slippage = pd.concat([self.slippage, pd.DataFrame(s).T], axis=0)

            
        
        #SELL M1 - BUY M2
        #Sell when the signal price delta adjusted for spread and fees is above the predicted autoregressive mean and no long exposure is outstanding
        elif (((d - spread_impact) > self.means[self.i]) and self.out_trades.empty) or (((d - spread_impact) > self.means[self.i]) and (self.out_trades['qty'] * self.out_trades['side']).sum() < 0):

            if self.trade_right: #Is it allowed to trade the right side of the distribution as per model's specs ?

                beven_z_score = ((d - fee) - self.means[i_next]) / self.vols[i_next] #Breakeven Z-Value at predicted t+1

                if self.extremes_left and (abs(beven_z_score) > self.z_tail_left):
                    """
                    Under current extreme values the breakeven probability is the joint probability of being below the left extreme value on a Student-t distribution and the probability of being below the t+1 breakeven Z-Value on a GPD.
                    So fee_beven_prob = P(t_ext ∩ g_z)
                    With :
                        -t_ext ≈ 1-CDF(extreme_threshold~T(0, 1))
                        -g_z ≈ 1-CDF(|breakeven_z_score|~GPD(degrees_of_freedom, 0, 1))
                    """
                    fee_beven_prob = stats.t.sf(self.z_tail_left, df=self.d_f) * stats.genpareto.sf(abs(beven_z_score), self.gpd_l_shape, self.gpd_l_loc, self.gpd_l_scale)
                else:
                    #Otherwise fee_beven_prob ≈ CDF(breakeven_z_score~T(degrees_of_freedom, mean, volatility))
                    fee_beven_prob = stats.t.cdf(d-fee, loc=self.means[i_next], scale=self.vols[i_next], df=self.d_f)

                #Trade signal : Place order when the breakeven probability is above the required minimum
                if fee_beven_prob > self.MIN_PROBABILITY_ENTRY_SIGNAL:

                    #SIZING
                    #Calculate trade size according to volume constraints and risk apetite
                    #Leverage should be set to have MAX_LOSS_PROB probability of experiencing a loss of at least MAX_EQ_LOSS

                    #Compute the minimum losss in price delta at the MAX_LOSS probability

                    #Only when extreme value theory applies to the right tail of the distribution
                    if self.extremes_right:
                        thr_prob = stats.t.sf((self.z_tail_right), loc=0, scale=1, df=self.d_f) #Probability of ending in extreme values in the Student-t distribution
                        """
                        Under extreme values theory the maximum acceptable loss occurs at a price delta corresponding to the conditional probability of experiencing MAX_LOSS_PROB given the probability of being in the extreme right tail (proven that Z_max_loss > extreme_tail)
                        Therefore : `max_px_delta_loss` = -1/CDF:(quantile) ~ GPD(shape, gpd_mean, gpd_vol) * volatility - mean
                        With :
                            - quantile = 1 - P(MAX_LOSS | P(extreme_tail)) = 1 - [MAX_LOSS / CDF(-extreme_tail) ~ T(degrees_of_freedom, 0, 1)]
                        """
                        max_px_delta_loss = (-stats.genpareto.ppf(1 - (self.MAX_LOSS_PROB / thr_prob), self.gpd_r_shape, loc=self.gpd_r_loc, scale=self.gpd_r_scale) * self.vols[i_next] - abs(self.means[i_next]))
                    #Otherwise
                    else:
                        #`max_px_delta_loss` = 1/CDF:(quantile) ~ T(degrees_of_freeddom, mean, volatility)
                        max_px_delta_loss = stats.t.ppf(self.MAX_LOSS_PROB, loc=self.means[i_next], scale=self.vols[i_next], df=self.d_f)

                    max_px_deviation = max_px_delta_loss * mid_m1 / 1e4 #Maximum acceptable deviation in bps

                    #Maximum leverage is the leverage at which MAX_EQ_LOSS of EQUITY is lost given the maximum acceptable deviation in bps
                    #As equity_loss = (loss_ratio * EQUITY), the maximum size is equity_loss / (p_t - p_t-1)
                    size = (-self.MAX_EQ_LOSS * self.EQUITY) / max_px_deviation

                    qty = min(self.depth_m1.best_bid_qty, abs(size))
                    
                    #Fire order
                    self.hbt.submit_sell_order(0, order_id, self.depth_m1.best_bid, qty, GTC, MARKET, True) #M1
                    self.hbt.submit_buy_order(1, order_id, self.depth_m2.best_ask, qty, GTC, MARKET, True) #M2

                    exec_px_m1 = self.hbt.orders(0).get(order_id).price
                    exec_px_m2 = self.hbt.orders(1).get(order_id).price
                    exec_delta = ((exec_px_m1 - exec_px_m2)/exec_px_m1*1e4)#Observed price delta at execution

                    #Hydrate outstanding trades
                    new_entry = pd.Series(
                            [
                            order_id,
                            exec_px_m1,
                            exec_px_m2,
                            exec_delta,
                            abs(self.hbt.orders(0).get(order_id).qty),
                            -1.0,
                            fee,
                            self.means[self.i],
                            np.nan,
                            np.nan
                            ],
                            index=self.out_cols
                        )
                    self.out_trades = pd.concat([self.out_trades, pd.DataFrame(new_entry).T.set_index('order_id')], axis=0)

                    self._print_orders(order_cat="Entry", order_id=order_id)

                    #Hydrate slippage
                    s = pd.Series(
                        [
                            mid_m1,
                            self.depth_m1.best_bid,
                            mid_m2,
                            self.depth_m2.best_ask,
                            d*mid_m1/1e4,
                            (self.depth_m1.best_ask - self.depth_m2.best_bid),
                            (((self.depth_m1.best_bid + self.depth_m1.best_ask)/2) - (self.depth_m1.best_bid)),
                            ((self.depth_m2.best_ask) - ((self.depth_m2.best_bid + self.depth_m2.best_ask)/2)),
                        ],
                        index=self.sl_cols
                    )
                    self.slippage = pd.concat([self.slippage, pd.DataFrame(s).T], axis=0)

    def _stop_loss(self):
        """
        Cut the losses by firing orders to flatten exposure on trades that are deemed not profitable given the market current conditions and strategy parameters.
        """

        """
        #Case 1 : Executed price delta does not recover fees
        case_1 = self.out_trades.loc[abs(self.out_trades['px_delta']) < self.out_trades['fee'], ['qty', 'side', 'exec_px_m1', 'exec_px_m2']]
        if not case_1.empty :
            qty = (case_1['qty'] * case_1['side']).sum()
            if qty > 0:
                hbt.submit_sell_order(0, i, depth_m1.best_bid, abs(qty), GTC, MARKET, True)
                hbt.submit_buy_order(1, i, depth_m2.best_ask, abs(qty), GTC, MARKET, True)
            else:
                hbt.submit_buy_order(0, i, depth_m1.best_ask, abs(qty), GTC, MARKET, True)
                hbt.submit_sell_order(1, i, depth_m2.best_bid, abs(qty), GTC, MARKET, True)
            self.out_trades = self.out_trades.drop(case_1.index)
            print_orders(order_cat="Stop loss", order_id=i, px_init_m1=case_1['exec_px_m1'].to_list(), px_init_m2=case_1['exec_px_m2'].to_list())
            return 1
            """
        
        #Case 2 : The expected probabilty of recovering fees in the next interval is below the minimum threshold
        case_2 = self.out_trades.loc[self.out_trades['fee_recover_prob'] < self.MAX_PROBABILITY_STOP_LOSS_SIGNAL, ['qty', 'side', 'exec_px_m1', 'exec_px_m2']]
        if not case_2.empty:
            qty = (case_2['qty'] * case_2['side']).sum()
            if qty > 0:
                self.hbt.submit_sell_order(0, self.i, self.depth_m1.best_bid, abs(qty), GTC, MARKET, True)
                self.hbt.submit_buy_order(1, self.i, self.depth_m2.best_ask, abs(qty), GTC, MARKET, True)
            else:
                self.hbt.submit_buy_order(0, self.i, self.depth_m1.best_ask, abs(qty), GTC, MARKET, True)
                self.hbt.submit_sell_order(1, self.i, self.depth_m2.best_bid, abs(qty), GTC, MARKET, True)
            self.out_trades = self.out_trades.drop(case_2.index) #Matching trades from the entry event are not outstanding anynore
            self._print_orders(order_cat="Stop loss", order_id=self.i, px_init_m1=case_2['exec_px_m1'].to_list(), px_init_m2=case_2['exec_px_m2'].to_list())
            return 1

        #Case 3 : The price delta shifted in the wrong direction between the signal and the actual execution
        case_3 = self.out_trades.loc[((self.out_trades['px_delta'] - self.out_trades['mean']) * self.out_trades['side']) > 0] #Correct direction will always give a negative result
        if not case_3.empty:
            qty = (case_3['qty'] * case_3['side']).sum()
            if qty > 0:
                self.hbt.submit_sell_order(0, self.i, self.depth_m1.best_bid, abs(qty), GTC, MARKET, True)
                self.hbt.submit_buy_order(1, self.i, self.depth_m2.best_ask, abs(qty), GTC, MARKET, True)
            else:
                self.hbt.submit_buy_order(0, self.i, self.depth_m1.best_ask, abs(qty), GTC, MARKET, True)
                self.hbt.submit_sell_order(1, self.i, self.depth_m2.best_bid, abs(qty), GTC, MARKET, True)
            self.out_trades = self.out_trades.drop(case_3.index) #Matching trades from the entry event are not outstanding anynore
            self._print_orders(order_cat="Stop loss", order_id=self.i, px_init_m1=case_3['exec_px_m1'].to_list(), px_init_m2=case_3['exec_px_m2'].to_list())
            return 1                  
        return 0
        
    def _take_profits(self, d:float):
        """
        Harvest profits by firing orders to flatten exposure on trades that hit the price delta target.

        Args:
            - d : Signal price delta
        """

        order_id = self.sample_size * 3 + self.i

        #Target price is crossed
        cash_out_long = self.out_trades.loc[(d >= self.out_trades['delta_target']) & (self.out_trades['side'] == 1),:]
        cash_out_short = self.out_trades.loc[(d <= self.out_trades['delta_target']) & (self.out_trades['side'] == -1),:]
        if not cash_out_long.empty:
            abs_net_exposure = abs((cash_out_long['qty'] * cash_out_long['side']).sum())
            self.hbt.submit_sell_order(0, order_id, self.depth_m1.best_bid, abs_net_exposure, GTC, MARKET, True)
            self.hbt.submit_buy_order(1, order_id, self.depth_m2.best_ask, abs_net_exposure, GTC, MARKET, True)
            self.out_trades = self.out_trades.drop(cash_out_long.index)
            self._print_orders(order_cat="Take profits", order_id=order_id, px_init_m1=cash_out_long['exec_px_m1'].to_list(), px_init_m2=cash_out_long['exec_px_m2'].to_list())
        elif not cash_out_short.empty:
            abs_net_exposure = abs((cash_out_short['qty'] * cash_out_short['side']).sum())
            self.hbt.submit_buy_order(0, order_id, self.depth_m1.best_ask, abs_net_exposure, GTC, MARKET, True)
            self.hbt.submit_sell_order(1, order_id, self.depth_m2.best_bid, abs_net_exposure, GTC, MARKET, True)
            self.out_trades = self.out_trades.drop(cash_out_short.index)
            self._print_orders(order_cat="Take profits", order_id=order_id, px_init_m1=cash_out_short['exec_px_m1'].to_list(), px_init_m2=cash_out_short['exec_px_m2'].to_list())

    def _backtest(self, px_delta:pd.Series, mid_quote:pd.DataFrame):
        """
        Backtester loop

        Args:
            - px_delta : Price deltas at signal time
            - mid_quote : M1 and M2 mid prices at signal time
        """

        #Instantiate slippage DataFrame
        self.sl_cols = [
            'signal_px_m1',
            'executed_px_m1',
            'signal_px_m2',
            'executed_px_m2',
            'signal_delta',
            'actual_delta',
            'spread_impact_m1',
            'spread_impact_m2'
        ]
        self.slippage = pd.DataFrame(columns=self.sl_cols)

        #Instantiate the outstanding trades DataFrame
        self.out_cols = ['order_id', 'exec_px_m1', 'exec_px_m2', 'px_delta', 'qty', 'side' ,'fee', 'mean', 'delta_target', 'fee_recover_prob']
        self.out_trades = pd.DataFrame(columns=self.out_cols).set_index('order_id')

        last_spread_m1 = 0.0
        last_spread_m2 = 0.0

        #Systematic decisions at every time interval
        self.i = 1
        while self.hbt.elapse(self._second_to_precision(self.STEP_TIME_INTEVAL, self.TIMESTAMP_PRECISION)) == 0:
            self._update_pred_stats()
            d = px_delta.iloc[self.i] #Price delta at signal time
            mid_quotes = (mid_quote['mid_m1'].iloc[self.i], mid_quote['mid_m2'].iloc[self.i]) #M1 and M2 mid prices at signal time
            self.depth_m1 = self.hbt.depth(0)
            self.depth_m2 = self.hbt.depth(1)

            self._stop_loss()
            self._take_profits(d=d)
            self._entry_exec(d=d, mid_quotes=mid_quotes, last_spread_m1=last_spread_m1, last_spread_m2=last_spread_m2)

            self.recorder.recorder.record(self.hbt)

            #Record spreads at time t for usage at t+1
            last_spread_m1 = (self.depth_m1.best_ask - self.depth_m1.best_bid) / 2
            last_spread_m2 = (self.depth_m2.best_ask - self.depth_m2.best_bid) / 2

            self.i += 1

        self.hbt.close()

        #Export results and spreads
        bpath = self._std_basepath()
        rpath = bpath + f"hftrecords/{self.comb}.npz"
        self.recorder.to_npz(rpath)
        self._log_it(f"Saved records : {rpath}", print_comb=True)
        spath = bpath + f"trades_slippage/{self.comb}_slippage.csv"
        self.slippage.to_csv(spath)
        self._log_it(f"Saved trades slippage : {spath}", print_comb=True)

        #Return the path of the saved records as Recorder object cannot be used directly for building statistics
        return {
            self.comb : rpath
        }
