import numpy as np
from json import loads
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest


def asset_const(
    # --- Required ---
    dataset,
    exchange,
    instrument,
    # --- Required with defaults ---
    linear_asset_multiplier=1.0,
    queue_model='risk_adverse',        # 'risk_adverse' | 'prob'
    exchange_model='no_partial_fill',  # 'no_partial_fill' | 'partial_fill'
    # --- Optional ---
    dataset_snap=None,
    latency_order=None,
    latency_feed=None,
    fee_model=None,                    # 'trading_value' | 'trading_qty' | 'flat_per_trade'
    fee_maker=None,
    fee_taker=None,
    last_trades_capacity=None,
):
    """HFTBACKTEST ASSET CONSTRUCTOR"""

    #Get exhange and intsrument details
    with open('./exchanges_details.json', 'r') as f:
        exch_dtls = loads(f.read()) 
    
    asset = BacktestAsset()

    # --- Required ---
    asset = asset.data(dataset)
    asset = asset.linear_asset(linear_asset_multiplier)
    asset = asset.tick_size(exch_dtls[exchange][instrument]['tick_size'])
    asset = asset.lot_size(exch_dtls[exchange][instrument]['lot'])

    # --- Required (one-of): queue model ---
    if queue_model == 'risk_adverse':
        asset = asset.risk_adverse_queue_model()
    elif queue_model == 'prob':
        asset = asset.prob_queue_model()
    else:
        raise ValueError(
            f"Unknown queue_model '{queue_model}'. Use 'risk_adverse' or 'prob'."
        )

    # --- Required (one-of): exchange model ---
    if exchange_model == 'no_partial_fill':
        asset = asset.no_partial_fill_exchange()
    elif exchange_model == 'partial_fill':
        asset = asset.partial_fill_exchange()
    else:
        raise ValueError(
            f"Unknown exchange_model '{exchange_model}'. Use 'no_partial_fill' or 'partial_fill'."
        )

    # --- Optional ---
    if dataset_snap is not None:
        asset = asset.initial_snapshot(dataset_snap)

    if latency_order is not None and latency_feed is not None:
        asset = asset.constant_latency(latency_order, latency_feed)

    if fee_model is not None:
        if fee_model == 'trading_value':
            asset = asset.trading_value_fee_model(fee_maker, fee_taker)
        elif fee_model == 'trading_qty':
            asset = asset.trading_qty_fee_model(fee_maker, fee_taker)
        elif fee_model == 'flat_per_trade':
            asset = asset.flat_per_trade_fee_model(fee_maker, fee_taker)
        else:
            raise ValueError(
                f"Unknown fee_model '{fee_model}'. "
                "Use 'trading_value', 'trading_qty', or 'flat_per_trade'."
            )

    if last_trades_capacity is not None:
        asset = asset.last_trades_capacity(last_trades_capacity)

    return asset