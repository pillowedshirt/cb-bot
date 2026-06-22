import csv, json, math, os, time, traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from runtime_paths import ensure_runtime_dirs, runtime_path, write_generated_file_meta

try:
    from historical_data_provider import BinanceBulkHistoricalProvider, write_normalized_candles_to_bot_cache
except Exception:
    BinanceBulkHistoricalProvider = None
    write_normalized_candles_to_bot_cache = None

CONTINUOUS_RESEARCH_HISTORY_COLUMNS = ["ts","dt_utc","cycle_id","cycle_type","status","duration_sec","input_rows","output_rows","reason"]
RESEARCH_FILE_HEALTH_COLUMNS = ["ts","dt_utc","filename","path","exists","rows","products","health","priority","reason"]
RESEARCH_BACKFILL_PLAN_COLUMNS = ["ts","dt_utc","task_id","task_type","product_id","timeframe","priority","status","reason"]
BACKGROUND_REPLAY_EXPANSION_COLUMNS = ["ts","dt_utc","cycle_id","task_id","product_id","timeframe","status","rows_written","start_ts","end_ts","reason"]
MARKET_STATE_ANALOG_MATCH_COLUMNS = ["ts","dt_utc","cycle_id","product_id","match_rank","similarity_score","source_timeframe","source_row_ts","source_regime","outcome_bps","features_used","reason"]
MARKET_STATE_ANALOG_SUMMARY_COLUMNS = ["ts","dt_utc","cycle_id","product_id","analog_sample_count","analog_avg_outcome_bps","analog_median_outcome_bps","analog_win_rate","analog_p25_bps","analog_p75_bps","analog_best_similarity","analog_gate","size_multiplier","reason"]
SELL_MODEL_RATIO_GRID_COLUMNS = ["ts","dt_utc","cycle_id","product_id","market_state_bucket","sample_count","scalp_target_mult","core_target_mult","scalp_pullback_pct","core_pullback_pct","max_hold_minutes","avg_net_bps","median_net_bps","win_rate","p25_net_bps","p75_net_bps","avg_hold_minutes","consistency_score","reason"]
ADAPTIVE_SELL_MODEL_POLICY_COLUMNS = ["ts","dt_utc","cycle_id","product_id","market_state_bucket","sample_count","scalp_target_mult","core_target_mult","scalp_pullback_pct","core_pullback_pct","max_hold_minutes","expected_avg_net_bps","expected_win_rate","consistency_score","policy_confidence","policy_gate","reason"]
ADAPTIVE_DECISION_POLICY_COLUMNS = ["ts","dt_utc","cycle_id","product_id","market_state_bucket","sample_count","buy_score_delta","probability_delta","ev_delta_bps","position_size_multiplier","scalp_target_mult","core_target_mult","scalp_pullback_pct","core_pullback_pct","max_hold_minutes","policy_confidence","policy_gate","reason"]

CROSS_ASSET_ANALOG_MATCH_COLUMNS = ["ts","dt_utc","cycle_id","target_product_id","source_product_id","match_rank","similarity_score","transfer_weight","weighted_similarity","source_timeframe","source_row_ts","source_regime","source_time_bucket","outcome_bps","weighted_outcome_bps","features_used","reason"]
CROSS_ASSET_ANALOG_SUMMARY_COLUMNS = ["ts","dt_utc","cycle_id","target_product_id","analog_sample_count","same_product_sample_count","cross_product_sample_count","weighted_avg_outcome_bps","weighted_median_outcome_bps","weighted_win_rate","weighted_p25_bps","weighted_p75_bps","best_similarity","avg_transfer_weight","analog_gate","size_multiplier","reason"]
CROSS_ASSET_SELL_MODEL_RATIO_GRID_COLUMNS = ["ts","dt_utc","cycle_id","target_product_id","market_state_bucket","sample_count","same_product_sample_count","cross_product_sample_count","scalp_target_mult","core_target_mult","scalp_pullback_pct","core_pullback_pct","max_hold_minutes","weighted_avg_net_bps","weighted_median_net_bps","weighted_win_rate","weighted_p25_net_bps","weighted_p75_net_bps","avg_hold_minutes","consistency_score","reason"]
CROSS_ASSET_ADAPTIVE_DECISION_POLICY_COLUMNS = ["ts","dt_utc","cycle_id","target_product_id","market_state_bucket","sample_count","same_product_sample_count","cross_product_sample_count","weighted_avg_net_bps","weighted_win_rate","weighted_p25_net_bps","buy_score_delta","probability_delta","ev_delta_bps","position_size_multiplier","scalp_target_mult","core_target_mult","scalp_pullback_pct","core_pullback_pct","max_hold_minutes","policy_confidence","policy_gate","reason"]

MAX_BUY_SCORE_DELTA=4.0; MAX_PROBABILITY_DELTA=0.035; MAX_EV_DELTA_BPS=6.0; MIN_POSITION_SIZE_MULT=0.35; MAX_POSITION_SIZE_MULT=1.0
MIN_POLICY_SAMPLE_COUNT=20; STRONG_POLICY_SAMPLE_COUNT=50

ENABLE_CROSS_ASSET_ANALOG_RESEARCH = True
CROSS_ASSET_MAX_MATCHES = 160
CROSS_ASSET_MIN_SAMPLE_COUNT = 25
CROSS_ASSET_MIN_SAME_PRODUCT_SAMPLE_FOR_FULL_TRUST = 20
SAME_PRODUCT_TRANSFER_WEIGHT = 1.00
CROSS_PRODUCT_BASE_TRANSFER_WEIGHT = 0.55
CROSS_PRODUCT_MIN_TRANSFER_WEIGHT = 0.25
CROSS_PRODUCT_MAX_TRANSFER_WEIGHT = 0.75
CROSS_ASSET_STRONG_POLICY_CONFIDENCE = 0.65
CROSS_ASSET_MIN_POLICY_CONFIDENCE = 0.25
TIME_BUCKET_MINUTES = 60
SELL_SCALP_TARGET_MULT_GRID=[0.45,0.55,0.65,0.75]; SELL_CORE_TARGET_MULT_GRID=[0.85,1.0,1.15,1.3]
SELL_SCALP_PULLBACK_GRID=[0.0010,0.0015,0.0020,0.0025,0.0030]; SELL_CORE_PULLBACK_GRID=[0.0015,0.0025,0.0035,0.0045,0.0060]; SELL_MAX_HOLD_MINUTES_GRID=[20,35,60,90]
LABEL_COLS={"product_id","market_regime","regime_tag","quant_volatility_cluster_state","session_liquidity_setup","structure_state","value_acceptance_state","volume_node_state","quant_boundary_state","timeframe","source_file"}

def _utc_ts(): return float(time.time())
def _utc_dt(ts_value=None): return datetime.fromtimestamp(_utc_ts() if ts_value is None else float(ts_value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
def _clip(v,lo,hi): return max(float(lo), min(float(hi), float(v)))
def _read_csv(path):
    try:
        return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path)>0 else pd.DataFrame()
    except Exception: return pd.DataFrame()
def _write_rows(path, cols, rows, reason):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); tmp=path+'.tmp'
    with open(tmp,'w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(cols); w.writerows(rows)
    os.replace(tmp,path); write_generated_file_meta(path, reason=reason)
def _append_rows(path, cols, rows, reason):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True); hdr=not os.path.exists(path) or os.path.getsize(path)<=0
    with open(path,'a',newline='',encoding='utf-8') as f:
        w=csv.writer(f); (w.writerow(cols) if hdr else None); w.writerows(rows)
    write_generated_file_meta(path, reason=reason)

def _latest_market_rows():
    m=_read_csv(runtime_path('market.csv'))
    if m.empty or 'product_id' not in m.columns: return pd.DataFrame()
    if 'ts' in m.columns: m['ts']=pd.to_numeric(m['ts'],errors='coerce').fillna(0); m=m.sort_values('ts')
    return m.groupby('product_id',as_index=False).tail(1)

def _historical_rows():
    frames=[]
    for fn in ['historical_shadow_replay.csv','candidate_replay.csv','trade_outcomes.csv','four_pass_sell_path_replay.csv']:
        f=_read_csv(runtime_path(fn))
        if f.empty or 'product_id' not in f.columns: continue
        if 'outcome_bps' not in f.columns:
            for c in ['realized_or_proxy_net_bps','binance_taker_taker_net_pnl_bps','binance_maker_taker_net_pnl_bps','net_pnl_bps','move_bps','buy_net_bps','realized_net_pnl_bps']:
                if c in f.columns: f['outcome_bps']=pd.to_numeric(f[c],errors='coerce'); break
        if 'outcome_bps' in f.columns:
            f=f.copy(); f['source_file']=fn; frames.append(f)
    if not frames: return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True); out['outcome_bps']=pd.to_numeric(out['outcome_bps'],errors='coerce')
    return out.dropna(subset=['outcome_bps'])

def _proportional_feature_frame(frame):
    if frame is None or frame.empty: return pd.DataFrame()
    out=frame.copy(); price=pd.Series(0.0,index=out.index)
    for c in ['price','mid','current_price','close','entry_price']:
        if c in out.columns: price=pd.to_numeric(out[c],errors='coerce').fillna(0); break
    feat=pd.DataFrame(index=out.index)
    for c in ['spread_bps','cost_bps','expected_net_edge_bps','projected_forward_gain_bps','momentum_1_bps','momentum_3_bps','momentum_5_bps','momentum_15_bps','quant_forecast_return_bps','quant_conditional_volatility_bps','poc_distance_bps','atr_bps','rsi','score','entry_score','probability','estimated_prob_up','calibrated_p_win','order_book_imbalance','liquidity_risk_score','relative_volume','volume_z','volume_profile_leader_buy_score','price_action_buy_score','market_structure_buy_score','quant_buy_score']:
        if c in out.columns: feat[c]=pd.to_numeric(out[c],errors='coerce').fillna(0)
    if 'spread_bps' not in feat.columns and {'bid','ask'}.issubset(out.columns):
        bid=pd.to_numeric(out['bid'],errors='coerce').fillna(0); ask=pd.to_numeric(out['ask'],errors='coerce').fillna(0); mid=(bid+ask)/2
        feat['spread_bps']=np.where(mid>0,((ask-bid)/mid)*10000,0)
    hi=next((c for c in ['range_high','session_high','high','recent_high'] if c in out.columns),None); lo=next((c for c in ['range_low','session_low','low','recent_low'] if c in out.columns),None)
    if hi and lo:
        h=pd.to_numeric(out[hi],errors='coerce').fillna(0); l=pd.to_numeric(out[lo],errors='coerce').fillna(0); width=(h-l).replace(0,np.nan)
        feat['range_position_0_1']=((price-l)/width).clip(0,1).fillna(.5); feat['range_width_bps']=np.where(price>0,((h-l)/price)*10000,0)
        feat['distance_to_range_high_bps']=np.where(price>0,((h/price)-1)*10000,0); feat['distance_to_range_low_bps']=np.where(price>0,((price/l.replace(0,np.nan))-1)*10000,0)
    if {'order_book_top_depth_usd','expected_order_notional_usd'}.issubset(out.columns):
        feat['depth_to_order_ratio']=(pd.to_numeric(out['order_book_top_depth_usd'],errors='coerce').fillna(0)/pd.to_numeric(out['expected_order_notional_usd'],errors='coerce').replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0).clip(0,100)
    for c in LABEL_COLS:
        if c in out.columns: feat[c]=out[c].astype(str)
    for c in ['ts','entry_ts','replay_ts']:
        if c in out.columns: feat['row_ts']=pd.to_numeric(out[c],errors='coerce').fillna(0); break
    if 'row_ts' not in feat.columns: feat['row_ts']=0.0
    try:
        row_ts = pd.to_numeric(feat.get('row_ts', 0.0), errors='coerce').fillna(0.0)
        dt_utc = pd.to_datetime(row_ts, unit='s', utc=True, errors='coerce')
        hour_float = dt_utc.dt.hour.fillna(0).astype(float) + dt_utc.dt.minute.fillna(0).astype(float) / 60.0
        feat['tod_sin'] = np.sin(2.0 * np.pi * hour_float / 24.0)
        feat['tod_cos'] = np.cos(2.0 * np.pi * hour_float / 24.0)
        feat['dayofweek_sin'] = np.sin(2.0 * np.pi * dt_utc.dt.dayofweek.fillna(0).astype(float) / 7.0)
        feat['dayofweek_cos'] = np.cos(2.0 * np.pi * dt_utc.dt.dayofweek.fillna(0).astype(float) / 7.0)
        feat['time_bucket_utc'] = dt_utc.dt.hour.fillna(0).astype(int).astype(str).str.zfill(2) + ':00'
    except Exception:
        feat['tod_sin'] = 0.0; feat['tod_cos'] = 1.0; feat['dayofweek_sin'] = 0.0; feat['dayofweek_cos'] = 1.0; feat['time_bucket_utc'] = 'unknown_time'
    if 'order_book_top_depth_usd' in out.columns:
        depth = pd.to_numeric(out['order_book_top_depth_usd'], errors='coerce').fillna(0.0)
        feat['log_depth_usd'] = np.log1p(depth.clip(lower=0.0))
    if 'quote_volume_24h' in out.columns:
        qv = pd.to_numeric(out['quote_volume_24h'], errors='coerce').fillna(0.0)
        feat['log_quote_volume_24h'] = np.log1p(qv.clip(lower=0.0))
    if {'order_book_top_depth_usd', 'quote_volume_24h'}.issubset(out.columns):
        depth = pd.to_numeric(out['order_book_top_depth_usd'], errors='coerce').fillna(0.0)
        qv = pd.to_numeric(out['quote_volume_24h'], errors='coerce').replace(0.0, np.nan)
        feat['depth_to_daily_volume_ratio'] = (depth / qv).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    for c in ['outcome_bps','max_favorable_bps','max_adverse_bps']:
        if c in out.columns: feat[c]=pd.to_numeric(out[c],errors='coerce').fillna(0)
    if 'held_seconds' in out.columns:
        feat['held_minutes'] = pd.to_numeric(out['held_seconds'], errors='coerce').fillna(0) / 60
    elif 'time_to_min_profit_minutes' in out.columns:
        feat['held_minutes'] = pd.to_numeric(out['time_to_min_profit_minutes'], errors='coerce').fillna(30)
    else:
        feat['held_minutes'] = pd.Series(30.0, index=out.index)
    return feat.replace([np.inf,-np.inf],np.nan)

def _feature_columns(cur,hist):
    forbidden={'outcome_bps','max_favorable_bps','max_adverse_bps','held_minutes','row_ts'}|LABEL_COLS
    return [c for c in cur.columns if c in hist.columns and c not in forbidden and pd.to_numeric(cur[c],errors='coerce').notna().any() and pd.to_numeric(hist[c],errors='coerce').notna().sum()>=10]

def _market_state_bucket(row):
    vals=[row.get('market_regime') or row.get('regime_tag') or row.get('quant_volatility_cluster_state') or 'unknown_regime', row.get('session_liquidity_setup') or 'unknown_session', row.get('structure_state') or 'unknown_structure', row.get('quant_boundary_state') or 'unknown_quant']
    return '|'.join(str(v).strip().lower().replace(' ','_') for v in vals)

def _analog_matches_for_product(product_id,current_row,hist_features,max_matches=80):
    ts=_utc_ts(); dt=_utc_dt(ts); cid=str(int(ts)); ph=hist_features[hist_features['product_id'].astype(str).eq(str(product_id))].copy()
    empty=[f'{ts:.6f}',dt,cid,product_id,0,0,0,0,0,0,0,'NO_ANALOGS',0.75,'no historical rows for product']
    if ph.empty: return [], empty, pd.DataFrame()
    cur=pd.DataFrame([current_row.to_dict()]); feats=_feature_columns(cur,ph)
    if not feats: empty[-3]='NO_COMMON_PROPORTIONAL_FEATURES'; empty[-1]='no shared proportional numeric features'; return [],empty,pd.DataFrame()
    dist=np.zeros(len(ph)); used=[]
    for f in feats:
        hv=pd.to_numeric(ph[f],errors='coerce'); cv=pd.to_numeric(cur[f],errors='coerce').iloc[0]
        if not math.isfinite(float(cv)): continue
        mean=float(hv.mean()); std=float(hv.std(ddof=0) or 0)
        if std>1e-12: dist+=np.square(((hv.fillna(mean)-mean)/std).to_numpy(dtype=float)-((float(cv)-mean)/std)); used.append(f)
    if not used: empty[-3]='NO_USABLE_PROPORTIONAL_FEATURES'; return [],empty,pd.DataFrame()
    ph['analog_distance']=np.sqrt(dist/max(1,len(used))); ph['similarity_score']=1/(1+ph['analog_distance']); ph=ph.sort_values('similarity_score',ascending=False).head(max_matches)
    outs=pd.to_numeric(ph['outcome_bps'],errors='coerce').fillna(0); rows=[]
    for rank,(_,r) in enumerate(ph.iterrows(),1): rows.append([f'{ts:.6f}',dt,cid,product_id,rank,f"{float(r.get('similarity_score',0)):.8f}",str(r.get('timeframe','')),r.get('row_ts',''),str(r.get('market_regime',r.get('regime_tag',''))),f"{float(r.get('outcome_bps',0)):.8f}",'|'.join(used),'proportional_market_state_analog_match'])
    n=len(outs); avg=float(outs.mean()) if n else 0; wr=float((outs>0).mean()) if n else 0
    gate='ANALOG_POSITIVE' if n>=20 and avg>5 and wr>=.55 else 'ANALOG_NEGATIVE' if n>=20 and (avg<-5 or wr<.42) else 'ANALOG_NEUTRAL' if n>=10 else 'ANALOG_LOW_SAMPLE'
    sm=1.0 if gate=='ANALOG_POSITIVE' else .5 if gate=='ANALOG_NEGATIVE' else .75 if gate=='ANALOG_NEUTRAL' else .65
    summary=[f'{ts:.6f}',dt,cid,product_id,n,f'{avg:.8f}',f'{float(outs.median()) if n else 0:.8f}',f'{wr:.8f}',f'{float(outs.quantile(.25)) if n else 0:.8f}',f'{float(outs.quantile(.75)) if n else 0:.8f}',f"{float(ph['similarity_score'].max()) if n else 0:.8f}",gate,f'{sm:.8f}',f"proportional_features={','.join(used)};max_matches={max_matches}"]
    return rows,summary,ph

def _weighted_mean(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    try:
        v = pd.to_numeric(values, errors='coerce')
        w = pd.to_numeric(weights, errors='coerce').fillna(0.0).clip(lower=0.0)
        mask = v.notna() & w.gt(0.0)
        if not mask.any(): return float(default)
        return float(np.average(v[mask], weights=w[mask]))
    except Exception:
        return float(default)

def _weighted_quantile(values: pd.Series, weights: pd.Series, q: float, default: float = 0.0) -> float:
    try:
        frame = pd.DataFrame({'v': pd.to_numeric(values, errors='coerce'), 'w': pd.to_numeric(weights, errors='coerce').fillna(0.0).clip(lower=0.0)}).dropna()
        frame = frame[frame['w'] > 0.0]
        if frame.empty: return float(default)
        frame = frame.sort_values('v')
        cdf = frame['w'].cumsum() / max(float(frame['w'].sum()), 1e-12)
        idx = int(np.searchsorted(cdf.to_numpy(), float(q), side='left'))
        return float(frame['v'].iloc[max(0, min(idx, len(frame)-1))])
    except Exception:
        return float(default)

def _weighted_win_rate(values: pd.Series, weights: pd.Series) -> float:
    try:
        wins = pd.to_numeric(values, errors='coerce').fillna(0.0).gt(0.0).astype(float)
        return _weighted_mean(wins, weights, default=0.0)
    except Exception:
        return 0.0

def _same_market_state_bonus(current_row: pd.Series, source_row: pd.Series) -> float:
    bonus = 0.0
    for col in ['market_regime','regime_tag','quant_volatility_cluster_state','session_liquidity_setup','structure_state','quant_boundary_state','time_bucket_utc']:
        cur = str(current_row.get(col, '')).strip().lower(); src = str(source_row.get(col, '')).strip().lower()
        if cur and src and cur == src: bonus += 0.04
    return min(0.25, bonus)

def _cross_asset_transfer_weight(*, target_product_id: str, source_product_id: str, current_row: pd.Series, source_row: pd.Series, similarity_score: float) -> float:
    if str(target_product_id) == str(source_product_id): return float(SAME_PRODUCT_TRANSFER_WEIGHT)
    base = float(CROSS_PRODUCT_BASE_TRANSFER_WEIGHT) + _same_market_state_bonus(current_row, source_row) + max(0.0, float(similarity_score) - 0.50) * 0.30
    return _clip(base, float(CROSS_PRODUCT_MIN_TRANSFER_WEIGHT), float(CROSS_PRODUCT_MAX_TRANSFER_WEIGHT))

def _cross_asset_analog_matches_for_target(target_product_id: str, current_row: pd.Series, hist_features: pd.DataFrame, max_matches: int = CROSS_ASSET_MAX_MATCHES) -> Tuple[List[List[Any]], List[Any], pd.DataFrame]:
    ts=_utc_ts(); dt=_utc_dt(ts); cid=str(int(ts))
    def empty(gate, reason): return [f'{ts:.6f}',dt,cid,target_product_id,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,gate,0.75,reason]
    if hist_features is None or hist_features.empty or 'product_id' not in hist_features.columns: return [], empty('NO_HISTORICAL_ROWS','no historical rows available across monitored products'), pd.DataFrame()
    cur=pd.DataFrame([current_row.to_dict()]); feats=_feature_columns(cur,hist_features)
    if not feats: return [], empty('NO_COMMON_CROSS_ASSET_FEATURES','no shared proportional numeric features across assets'), pd.DataFrame()
    local=hist_features.copy(); dist=np.zeros(len(local)); used=[]
    for f in feats:
        hv=pd.to_numeric(local[f],errors='coerce'); cv=pd.to_numeric(cur[f],errors='coerce').iloc[0]
        if not math.isfinite(float(cv)): continue
        mean=float(hv.mean()); std=float(hv.std(ddof=0) or 0.0)
        if std<=1e-12: continue
        dist += np.square(((hv.fillna(mean)-mean)/std).to_numpy(dtype=float)-((float(cv)-mean)/std)); used.append(f)
    if not used: return [], empty('NO_USABLE_CROSS_ASSET_FEATURES','features had unusable variance/current values'), pd.DataFrame()
    local['analog_distance']=np.sqrt(dist/max(1,len(used))); local['similarity_score']=1.0/(1.0+local['analog_distance'])
    local=local.sort_values('similarity_score',ascending=False).head(int(max_matches)).copy()
    local['transfer_weight']=[_cross_asset_transfer_weight(target_product_id=target_product_id, source_product_id=str(r.get('product_id','')), current_row=current_row, source_row=r, similarity_score=float(r.get('similarity_score',0) or 0)) for _,r in local.iterrows()]
    local['weighted_similarity']=pd.to_numeric(local['similarity_score'],errors='coerce').fillna(0.0)*local['transfer_weight']
    local['weighted_outcome_bps']=pd.to_numeric(local['outcome_bps'],errors='coerce').fillna(0.0)*local['transfer_weight']
    local=local.sort_values('weighted_similarity',ascending=False).head(int(max_matches)).copy()
    rows=[]
    for rank,(_,r) in enumerate(local.iterrows(),1):
        out=float(r.get('outcome_bps',0) or 0); tw=float(r.get('transfer_weight',0) or 0)
        rows.append([f'{ts:.6f}',dt,cid,target_product_id,str(r.get('product_id','')),rank,f"{float(r.get('similarity_score',0) or 0):.8f}",f'{tw:.8f}',f"{float(r.get('weighted_similarity',0) or 0):.8f}",str(r.get('timeframe','')),r.get('row_ts',''),str(r.get('market_regime',r.get('regime_tag',''))),str(r.get('time_bucket_utc','')),f'{out:.8f}',f'{out*tw:.8f}','|'.join(used),'cross_asset_proportional_market_state_analog_match'])
    outs=pd.to_numeric(local['outcome_bps'],errors='coerce').fillna(0.0); w=pd.to_numeric(local['transfer_weight'],errors='coerce').fillna(0.0).clip(lower=0.0)
    n=len(local); same=int(local['product_id'].astype(str).eq(str(target_product_id)).sum()); cross=n-same
    avg=_weighted_mean(outs,w); med=_weighted_quantile(outs,w,.5); wr=_weighted_win_rate(outs,w); p25=_weighted_quantile(outs,w,.25); p75=_weighted_quantile(outs,w,.75)
    gate='CROSS_ASSET_ANALOG_POSITIVE' if n>=CROSS_ASSET_MIN_SAMPLE_COUNT and avg>5 and wr>=.55 else 'CROSS_ASSET_ANALOG_NEGATIVE' if n>=CROSS_ASSET_MIN_SAMPLE_COUNT and (avg<-5 or wr<.42) else 'CROSS_ASSET_ANALOG_NEUTRAL' if n>=10 else 'CROSS_ASSET_ANALOG_LOW_SAMPLE'
    sm=1.0 if gate.endswith('POSITIVE') else .5 if gate.endswith('NEGATIVE') else .75 if gate.endswith('NEUTRAL') else .65
    summary=[f'{ts:.6f}',dt,cid,target_product_id,n,same,cross,f'{avg:.8f}',f'{med:.8f}',f'{wr:.8f}',f'{p25:.8f}',f'{p75:.8f}',f"{float(local['similarity_score'].max()) if n else 0:.8f}",f'{float(w.mean()) if n else 0:.8f}',gate,f'{sm:.8f}',f"cross_asset_proportional_features={','.join(used)};max_matches={max_matches};same={same};cross={cross}"]
    return rows, summary, local

def _simulate_cross_asset_sell_policy_on_analogs(*, cycle_id: str, target_product_id: str, market_state_bucket: str, analogs: pd.DataFrame) -> Tuple[List[List[Any]], Optional[List[Any]]]:
    ts=_utc_ts(); dt=_utc_dt(ts)
    if analogs is None or analogs.empty or len(analogs)<10: return [], None
    l=analogs.copy(); l['outcome_bps']=pd.to_numeric(l.get('outcome_bps',0.0),errors='coerce').fillna(0.0); l['max_favorable_bps']=pd.to_numeric(l.get('max_favorable_bps',l['outcome_bps']),errors='coerce').fillna(l['outcome_bps']); l['max_adverse_bps']=pd.to_numeric(l.get('max_adverse_bps',0.0),errors='coerce').fillna(0.0); l['held_minutes']=pd.to_numeric(l.get('held_minutes',30.0),errors='coerce').fillna(30.0); l['transfer_weight']=pd.to_numeric(l.get('transfer_weight',0.50),errors='coerce').fillna(0.50).clip(lower=0.05)
    n=len(l); same=int(l['product_id'].astype(str).eq(str(target_product_id)).sum()) if 'product_id' in l.columns else 0; cross=n-same; grid=[]; best=None; score=-1e18
    base=max(8.0,_weighted_quantile(l['max_favorable_bps'],l['transfer_weight'],.5)*.55); core=max(12.0,_weighted_quantile(l['max_favorable_bps'],l['transfer_weight'],.5)*.85)
    for sm in SELL_SCALP_TARGET_MULT_GRID:
      for cm in SELL_CORE_TARGET_MULT_GRID:
       for sp in SELL_SCALP_PULLBACK_GRID:
        for cp in SELL_CORE_PULLBACK_GRID:
         for mh in SELL_MAX_HOLD_MINUTES_GRID:
          vals=[]
          for _,r in l.iterrows():
            mfe=float(r.get('max_favorable_bps',0) or 0); mae=abs(float(r.get('max_adverse_bps',0) or 0)); raw=float(r.get('outcome_bps',0) or 0); held=float(r.get('held_minutes',30) or 30); pen=min(8.0,(held-float(mh))*.05) if held>float(mh) else 0.0; st=base*float(sm); ct=core*float(cm)
            vals.append(ct+max(0,mfe-ct)*max(.1,1-cp*120)-pen if mfe>=ct else st+max(0,mfe-st)*max(.05,1-sp*160)-pen if mfe>=st else raw-min(8,mae*.15)-pen)
          vals=pd.Series(vals,dtype='float64'); w=l['transfer_weight']; avg=_weighted_mean(vals,w); med=_weighted_quantile(vals,w,.5); wr=_weighted_win_rate(vals,w); p25=_weighted_quantile(vals,w,.25); p75=_weighted_quantile(vals,w,.75); ah=_weighted_mean(l['held_minutes'],w); cons=avg+med*.65+wr*18+p25*.85-max(0,-p25)*.65+min(.20,same/max(1,n))*8
          grid.append([f'{ts:.6f}',dt,cycle_id,target_product_id,market_state_bucket,n,same,cross,f'{sm:.8f}',f'{cm:.8f}',f'{sp:.8f}',f'{cp:.8f}',f'{float(mh):.6f}',f'{avg:.8f}',f'{med:.8f}',f'{wr:.8f}',f'{p25:.8f}',f'{p75:.8f}',f'{ah:.8f}',f'{cons:.8f}','cross_asset_sell_model_ratio_grid_test'])
          if cons>score: score=cons; best=dict(sample_count=n,same_count=same,cross_count=cross,weighted_avg=avg,weighted_median=med,weighted_win_rate=wr,weighted_p25=p25,weighted_p75=p75,avg_hold=ah,consistency=cons,scalp_target_mult=float(sm),core_target_mult=float(cm),scalp_pullback_pct=float(sp),core_pullback_pct=float(cp),max_hold_minutes=float(mh))
    if not best: return grid, None
    cr=best['cross_count']/max(1,best['sample_count']); sr=best['same_count']/max(1,best['sample_count'])
    conf=_clip((best['sample_count']/120)*.45+max(0,best['weighted_win_rate']-.45)*.80+max(0,best['weighted_p25'])/30+sr*.20-max(0,cr-.80)*.10,0,1)
    gate='CROSS_ASSET_POLICY_LOW_SAMPLE' if best['sample_count']<CROSS_ASSET_MIN_SAMPLE_COUNT else 'CROSS_ASSET_POLICY_DEFENSIVE' if best['weighted_p25']<-8 or best['weighted_win_rate']<.42 else 'CROSS_ASSET_POLICY_STRONG' if conf>=CROSS_ASSET_STRONG_POLICY_CONFIDENCE and best['weighted_avg']>5 else 'CROSS_ASSET_POLICY_NEUTRAL'
    bd,pdlt,ev,pm=(2,.015,3,.55) if gate=='CROSS_ASSET_POLICY_DEFENSIVE' else (-.75,-.003,-1,1.0) if gate=='CROSS_ASSET_POLICY_STRONG' else (0,0,0,.75 if best['sample_count']<50 else .9)
    policy=[f'{ts:.6f}',dt,cycle_id,target_product_id,market_state_bucket,best['sample_count'],best['same_count'],best['cross_count'],f"{best['weighted_avg']:.8f}",f"{best['weighted_win_rate']:.8f}",f"{best['weighted_p25']:.8f}",f'{_clip(bd,-MAX_BUY_SCORE_DELTA,MAX_BUY_SCORE_DELTA):.8f}',f'{_clip(pdlt,-MAX_PROBABILITY_DELTA,MAX_PROBABILITY_DELTA):.8f}',f'{_clip(ev,-MAX_EV_DELTA_BPS,MAX_EV_DELTA_BPS):.8f}',f'{_clip(pm,MIN_POSITION_SIZE_MULT,MAX_POSITION_SIZE_MULT):.8f}',f"{best['scalp_target_mult']:.8f}",f"{best['core_target_mult']:.8f}",f"{best['scalp_pullback_pct']:.8f}",f"{best['core_pullback_pct']:.8f}",f"{best['max_hold_minutes']:.6f}",f'{conf:.8f}',gate,f"cross_asset_adaptive_policy;same={best['same_count']};cross={best['cross_count']};avg={best['weighted_avg']:.2f};win={best['weighted_win_rate']:.3f};p25={best['weighted_p25']:.2f}"]
    return grid, policy

def _simulate_sell_policy_on_analogs(cycle_id,product_id,market_state_bucket,analogs):
    ts=_utc_ts(); dt=_utc_dt(ts)
    if analogs is None or analogs.empty or len(analogs)<10: return [],None,None
    l=analogs.copy(); l['outcome_bps']=pd.to_numeric(l.get('outcome_bps',0),errors='coerce').fillna(0); l['max_favorable_bps']=pd.to_numeric(l.get('max_favorable_bps',l['outcome_bps']),errors='coerce').fillna(l['outcome_bps']); l['max_adverse_bps']=pd.to_numeric(l.get('max_adverse_bps',0),errors='coerce').fillna(0); l['held_minutes']=pd.to_numeric(l.get('held_minutes',30),errors='coerce').fillna(30)
    n=len(l); grid=[]; best=None; best_score=-1e18; base=max(8,float(l['max_favorable_bps'].median()*.55)); core=max(12,float(l['max_favorable_bps'].median()*.85))
    for sm in SELL_SCALP_TARGET_MULT_GRID:
      for cm in SELL_CORE_TARGET_MULT_GRID:
       for sp in SELL_SCALP_PULLBACK_GRID:
        for cp in SELL_CORE_PULLBACK_GRID:
         for mh in SELL_MAX_HOLD_MINUTES_GRID:
          vals=[]
          for _,r in l.iterrows():
            mfe=float(r.max_favorable_bps); raw=float(r.outcome_bps); mae=abs(float(r.max_adverse_bps)); held=float(r.held_minutes); pen=min(8,(held-mh)*.05) if held>mh else 0
            vals.append((core*cm+max(0,mfe-core*cm)*max(.1,1-cp*120)-pen) if mfe>=core*cm else (base*sm+max(0,mfe-base*sm)*max(.05,1-sp*160)-pen) if mfe>=base*sm else raw-min(8,mae*.15)-pen)
          s=pd.Series(vals); avg=float(s.mean()); med=float(s.median()); wr=float((s>0).mean()); p25=float(s.quantile(.25)); p75=float(s.quantile(.75)); ah=float(l['held_minutes'].mean()); cons=avg+med*.65+wr*18+p25*.85-max(0,-p25)*.65
          grid.append([f'{ts:.6f}',dt,cycle_id,product_id,market_state_bucket,n,f'{sm:.8f}',f'{cm:.8f}',f'{sp:.8f}',f'{cp:.8f}',f'{float(mh):.6f}',f'{avg:.8f}',f'{med:.8f}',f'{wr:.8f}',f'{p25:.8f}',f'{p75:.8f}',f'{ah:.8f}',f'{cons:.8f}','sell_model_ratio_grid_proportional_analog_test'])
          if cons>best_score: best_score=cons; best=dict(sample_count=n,scalp_target_mult=sm,core_target_mult=cm,scalp_pullback_pct=sp,core_pullback_pct=cp,max_hold_minutes=mh,avg=avg,median=med,win_rate=wr,p25=p25,p75=p75,avg_hold=ah,consistency=cons)
    conf=_clip((n/80)*.55+max(0,best['win_rate']-.45)*.9+max(0,best['p25'])/25,0,1)
    gate='SELL_POLICY_LOW_SAMPLE' if n<MIN_POLICY_SAMPLE_COUNT else 'SELL_POLICY_DEFENSIVE' if best['p25']<-8 or best['win_rate']<.42 else 'SELL_POLICY_STRONG' if conf>=.65 and best['avg']>5 else 'SELL_POLICY_NEUTRAL'
    sell=[f'{ts:.6f}',dt,cycle_id,product_id,market_state_bucket,n,f"{best['scalp_target_mult']:.8f}",f"{best['core_target_mult']:.8f}",f"{best['scalp_pullback_pct']:.8f}",f"{best['core_pullback_pct']:.8f}",f"{best['max_hold_minutes']:.6f}",f"{best['avg']:.8f}",f"{best['win_rate']:.8f}",f"{best['consistency']:.8f}",f'{conf:.8f}',gate,f"best_sell_ratio_from_proportional_analogs;p25={best['p25']:.2f};median={best['median']:.2f};p75={best['p75']:.2f};avg_hold={best['avg_hold']:.2f}"]
    bd,pdlt,ev,pm=(2,.015,3,.55) if gate=='SELL_POLICY_DEFENSIVE' else (-1,-.005,-1.5,1.0) if gate=='SELL_POLICY_STRONG' else (0,0,0,.75 if n<STRONG_POLICY_SAMPLE_COUNT else .9)
    dec=[f'{ts:.6f}',dt,cycle_id,product_id,market_state_bucket,n,f'{_clip(bd,-MAX_BUY_SCORE_DELTA,MAX_BUY_SCORE_DELTA):.8f}',f'{_clip(pdlt,-MAX_PROBABILITY_DELTA,MAX_PROBABILITY_DELTA):.8f}',f'{_clip(ev,-MAX_EV_DELTA_BPS,MAX_EV_DELTA_BPS):.8f}',f'{_clip(pm,MIN_POSITION_SIZE_MULT,MAX_POSITION_SIZE_MULT):.8f}',f"{best['scalp_target_mult']:.8f}",f"{best['core_target_mult']:.8f}",f"{best['scalp_pullback_pct']:.8f}",f"{best['core_pullback_pct']:.8f}",f"{best['max_hold_minutes']:.6f}",f'{conf:.8f}',gate,'adaptive_decision_policy_from_sell_model_analog_research']
    return grid,sell,dec

def run_market_state_analog_research(max_matches=80):
    started=time.time(); ensure_runtime_dirs(); cur=_proportional_feature_frame(_latest_market_rows()); hist=_proportional_feature_frame(_historical_rows())
    if cur.empty or hist.empty: return {'status':'skipped','reason':f'current_empty={cur.empty};historical_empty={hist.empty}','matches':0,'summary_rows':0,'duration_sec':time.time()-started}
    matches=[]; sums=[]; grids=[]; sells=[]; decs=[]; cross_matches=[]; cross_sums=[]; cross_grids=[]; cross_decs=[]; cid=str(int(_utc_ts()))
    for _,r in cur.iterrows():
        pid=str(r.get('product_id',''))
        if not pid: continue
        rows,summary,af=_analog_matches_for_product(pid,r,hist,max_matches); matches+=rows; sums.append(summary)
        g,sell,dec=_simulate_sell_policy_on_analogs(cid,pid,_market_state_bucket(r),af); grids+=g
        if sell: sells.append(sell)
        if dec: decs.append(dec)
        if bool(ENABLE_CROSS_ASSET_ANALOG_RESEARCH):
            crows,csummary,caf=_cross_asset_analog_matches_for_target(pid,r,hist,max_matches=int(CROSS_ASSET_MAX_MATCHES)); cross_matches+=crows; cross_sums.append(csummary)
            cg,cp=_simulate_cross_asset_sell_policy_on_analogs(cycle_id=cid,target_product_id=pid,market_state_bucket=_market_state_bucket(r),analogs=caf); cross_grids+=cg
            if cp: cross_decs.append(cp)
    output_files=[
        ('market_state_analog_matches.csv',MARKET_STATE_ANALOG_MATCH_COLUMNS,matches,'same_product_proportional_market_state_analog_research'),
        ('market_state_analog_summary.csv',MARKET_STATE_ANALOG_SUMMARY_COLUMNS,sums,'same_product_proportional_market_state_analog_research'),
        ('sell_model_ratio_grid.csv',SELL_MODEL_RATIO_GRID_COLUMNS,grids,'same_product_background_sell_model_ratio_research'),
        ('adaptive_sell_model_policy.csv',ADAPTIVE_SELL_MODEL_POLICY_COLUMNS,sells,'same_product_adaptive_sell_model_policy'),
        ('adaptive_decision_policy.csv',ADAPTIVE_DECISION_POLICY_COLUMNS,decs,'same_product_adaptive_decision_policy'),
        ('cross_asset_analog_matches.csv',CROSS_ASSET_ANALOG_MATCH_COLUMNS,cross_matches,'cross_asset_proportional_market_state_analog_research'),
        ('cross_asset_analog_summary.csv',CROSS_ASSET_ANALOG_SUMMARY_COLUMNS,cross_sums,'cross_asset_proportional_market_state_analog_research'),
        ('cross_asset_sell_model_ratio_grid.csv',CROSS_ASSET_SELL_MODEL_RATIO_GRID_COLUMNS,cross_grids,'cross_asset_sell_model_ratio_research'),
        ('cross_asset_adaptive_decision_policy.csv',CROSS_ASSET_ADAPTIVE_DECISION_POLICY_COLUMNS,cross_decs,'cross_asset_adaptive_decision_policy'),
    ]
    for fn,cols,rows,reason in output_files: _write_rows(runtime_path(fn),cols,rows,reason)
    return {'status':'ok','reason':'same_product_and_cross_asset_proportional_research_completed','same_product_matches':len(matches),'same_product_summary_rows':len(sums),'same_product_sell_grid_rows':len(grids),'same_product_policy_rows':len(decs),'cross_asset_matches':len(cross_matches),'cross_asset_summary_rows':len(cross_sums),'cross_asset_sell_grid_rows':len(cross_grids),'cross_asset_policy_rows':len(cross_decs),'duration_sec':time.time()-started}

def _file_health(filename,min_rows=1,min_products=0):
    path=runtime_path(filename); r={'filename':filename,'path':path,'exists':os.path.exists(path),'rows':0,'products':0,'health':'MISSING','priority':100,'reason':'missing'}
    try:
        if not r['exists'] or os.path.getsize(path)<=0: return r
        if filename.endswith('.json'): r.update(health='OK',priority=20,reason='json_exists'); return r
        f=_read_csv(path)
        if f.empty: r.update(health='EMPTY',priority=90,reason='empty_or_header_only'); return r
        r['rows']=len(f); r['products']=int(f['product_id'].astype(str).nunique()) if 'product_id' in f.columns else 0
        if r['rows']<min_rows: r.update(health='LOW_ROWS',priority=80,reason=f"rows={r['rows']};min_rows={min_rows}")
        elif min_products and r['products']<min_products: r.update(health='LOW_PRODUCT_COVERAGE',priority=75,reason=f"products={r['products']};min_products={min_products}")
        else: r.update(health='OK',priority=10,reason='usable')
    except Exception as e: r.update(health='READ_ERROR',priority=95,reason=f'read_error:{e}')
    return r

def build_research_data_plan(products):
    ts=_utc_ts(); dt=_utc_dt(ts); health=[]; plan=[]; products=[str(p) for p in products]
    targets=[('historical_replay_15m_90d.csv',300*max(1,len(products)),max(1,len(products))),('historical_replay_1h_365d.csv',120*max(1,len(products)),max(1,len(products))),('historical_replay_1d_2y.csv',180*max(1,len(products)),max(1,len(products))),('historical_shadow_replay.csv',500,1),('candidate_replay.csv',500,1),('market_state_analog_summary.csv',1,1),('sell_model_ratio_grid.csv',1,1),('adaptive_sell_model_policy.csv',1,1),('adaptive_decision_policy.csv',1,1),('cross_asset_analog_summary.csv',1,1),('cross_asset_analog_matches.csv',1,1),('cross_asset_sell_model_ratio_grid.csv',1,1),('cross_asset_adaptive_decision_policy.csv',1,1)]
    for fn,mr,mp in targets:
        h=_file_health(fn,mr,mp); health.append([f'{ts:.6f}',dt,h['filename'],h['path'],bool(h['exists']),int(h['rows']),int(h['products']),h['health'],int(h['priority']),h['reason']])
    raw15=_read_csv(runtime_path('historical_replay_15m_90d.csv')); raw1h=_read_csv(runtime_path('historical_replay_1h_365d.csv')); raw1d=_read_csv(runtime_path('historical_replay_1d_2y.csv')); pol=_read_csv(runtime_path('adaptive_sell_model_policy.csv')); cross_pol=_read_csv(runtime_path('cross_asset_adaptive_decision_policy.csv'))
    def cnt(f,p): return 0 if f.empty or 'product_id' not in f.columns else int(f[f['product_id'].astype(str).eq(p)].shape[0])
    for p in products:
        for rows,tf,pri,why in [(cnt(raw15,p),'primary_15m_90d',90,'15m proportional analog rows low'),(cnt(raw1h,p),'regime_1h_365d',95,'1h regime analog rows low'),(cnt(raw1d,p),'daily_1d_2y',85,'daily structure rows low')]:
            lim=600 if '15m' in tf else 240 if '1h' in tf else 365
            if rows<lim: plan.append([f'{ts:.6f}',dt,f'{p}__expand_{tf}','expand_historical_cache',p,tf,pri,'planned',f'{why}: {rows}'])
        pn=0
        if not pol.empty and {'product_id','sample_count'}.issubset(pol.columns):
            sub=pol[pol['product_id'].astype(str).eq(p)]; pn=int(float(sub.tail(1)['sample_count'].iloc[0] or 0)) if not sub.empty else 0
        if pn<MIN_POLICY_SAMPLE_COUNT: plan.append([f'{ts:.6f}',dt,f'{p}__sell_model_ratio_research','sell_model_ratio_research',p,'current_state_analogs',100,'planned',f'sell policy sample count low: {pn}'])
        cpn=0
        if not cross_pol.empty and {'target_product_id','sample_count'}.issubset(cross_pol.columns):
            sub=cross_pol[cross_pol['target_product_id'].astype(str).eq(p)]; cpn=int(float(sub.tail(1)['sample_count'].iloc[0] or 0)) if not sub.empty else 0
        if cpn<CROSS_ASSET_MIN_SAMPLE_COUNT: plan.append([f'{ts:.6f}',dt,f'{p}__cross_asset_sell_model_research','cross_asset_sell_model_ratio_research',p,'all_coin_analogs',105,'planned',f'cross-asset sell policy sample count low: {cpn}'])
    plan.sort(key=lambda x:int(x[6]), reverse=True); _write_rows(runtime_path('research_file_health.csv'),RESEARCH_FILE_HEALTH_COLUMNS,health,'research_file_health_scan'); _write_rows(runtime_path('research_backfill_plan.csv'),RESEARCH_BACKFILL_PLAN_COLUMNS,plan,'sell_model_focused_research_backfill_plan')
    return {'health_rows':len(health),'plan_rows':len(plan),'top_tasks':plan[:10]}

def execute_one_background_backfill_task(products):
    started=time.time(); ts=_utc_ts(); dt=_utc_dt(ts); cid=str(int(ts))
    if BinanceBulkHistoricalProvider is None or write_normalized_candles_to_bot_cache is None: return {'status':'skipped','reason':'historical provider unavailable','duration_sec':time.time()-started}
    plan=_read_csv(runtime_path('research_backfill_plan.csv'))
    if plan.empty or 'task_type' not in plan.columns: return {'status':'skipped','reason':'no research_backfill_plan rows','duration_sec':time.time()-started}
    plan=plan[plan['task_type'].astype(str).eq('expand_historical_cache')].copy()
    if plan.empty: return {'status':'skipped','reason':'no expand_historical_cache tasks','duration_sec':time.time()-started}
    plan['priority']=pd.to_numeric(plan['priority'],errors='coerce').fillna(0); task=plan.sort_values('priority',ascending=False).iloc[0].to_dict(); pid=str(task.get('product_id','')); tf=str(task.get('timeframe','')); end=int(time.time())
    if tf=='primary_15m_90d': start=end-180*86400; out='historical_replay_15m_90d.csv'
    elif tf=='regime_1h_365d': start=end-730*86400; out='historical_replay_1h_365d.csv'
    elif tf=='daily_1d_2y': start=end-4*365*86400; out='historical_replay_1d_2y.csv'
    else: return {'status':'skipped','reason':f'unsupported timeframe {tf}','duration_sec':time.time()-started}
    try:
        provider=BinanceBulkHistoricalProvider(base_dir=os.path.dirname(runtime_path(out))); candles,info=provider.fetch_bulk_candles(product_id=pid,timeframe=tf,start_ts=start,end_ts=end); rows=write_normalized_candles_to_bot_cache(path=runtime_path(out),product_id=pid,candles=candles,min_ts=start)
        _append_rows(runtime_path('background_replay_expansion_summary.csv'),BACKGROUND_REPLAY_EXPANSION_COLUMNS,[[f'{ts:.6f}',dt,cid,str(task.get('task_id','')),pid,tf,'ok',int(rows),start,end,f'provider_info={info}']],'background_replay_expansion_summary')
        return {'status':'ok','task_id':str(task.get('task_id','')),'product_id':pid,'timeframe':tf,'rows_written':int(rows),'duration_sec':time.time()-started}
    except Exception as e:
        _append_rows(runtime_path('background_replay_expansion_summary.csv'),BACKGROUND_REPLAY_EXPANSION_COLUMNS,[[f'{ts:.6f}',dt,cid,str(task.get('task_id','')),pid,tf,'error',0,start,end,f'error={e}']],'background_replay_expansion_error')
        return {'status':'error','task_id':str(task.get('task_id','')),'error':str(e),'traceback':traceback.format_exc(),'duration_sec':time.time()-started}

def run_continuous_research_cycle():
    started=time.time(); ts=_utc_ts(); cid=str(int(ts))
    try:
        cur=_latest_market_rows(); products=cur['product_id'].dropna().astype(str).unique().tolist() if cur is not None and not cur.empty and 'product_id' in cur.columns else []
        data_plan=build_research_data_plan(products); expansion=execute_one_background_backfill_task(products); analog=run_market_state_analog_research(max_matches=80)
        status={'ts':ts,'dt_utc':_utc_dt(ts),'cycle_id':cid,'status':'ok','data_plan':data_plan,'expansion':expansion,'analog':analog,'duration_sec':time.time()-started}
        with open(runtime_path('continuous_research_status.json'),'w',encoding='utf-8') as f: json.dump(status,f,indent=2,sort_keys=True)
        write_generated_file_meta(runtime_path('continuous_research_status.json'),reason='continuous_research_cycle')
        _append_rows(runtime_path('continuous_research_history.csv'),CONTINUOUS_RESEARCH_HISTORY_COLUMNS,[[f'{ts:.6f}',_utc_dt(ts),cid,'proportional_analog_sell_model_research',str(analog.get('status','')),f"{float(status.get('duration_sec',0)):.6f}",0,int(analog.get('decision_policy_rows',0) or 0),str(analog.get('reason',''))]],'continuous_research_history_append')
        return status
    except Exception as e:
        status={'ts':ts,'dt_utc':_utc_dt(ts),'cycle_id':cid,'status':'error','error':str(e),'traceback':traceback.format_exc(),'duration_sec':time.time()-started}
        try:
            with open(runtime_path('continuous_research_status.json'),'w',encoding='utf-8') as f: json.dump(status,f,indent=2,sort_keys=True)
        except Exception: pass
        return status
