#!/usr/bin/env python3
"""
Compass Data Fetcher
Obtiene datos de opciones y calcula IV Rank y Skew Rank para el Compass
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Lista de símbolos a analizar
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WFC', 'DIS', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'TLT', 'COIN', 'PLTR', 'SOFI', 'UBER'
]

# Configuración
LOOKBACK_DAYS = 252  # 1 año para calcular percentiles
MIN_OPTIONS_REQUIRED = 5  # Mínimo de opciones para considerar el símbolo


def calculate_iv_rank(current_iv: float, historical_ivs: List[float]) -> float:
    """
    Calcula el IV Rank: percentil de la IV actual en los últimos 252 días
    
    Args:
        current_iv: Volatilidad implícita actual
        historical_ivs: Lista de IVs históricas
    
    Returns:
        IV Rank como porcentaje (0-100)
    """
    if not historical_ivs or current_iv is None:
        return 50.0  # Valor neutral si no hay datos
    
    historical_ivs = [iv for iv in historical_ivs if iv is not None]
    if len(historical_ivs) < 10:
        return 50.0
    
    # Calcular percentil
    rank = np.percentile(historical_ivs, 100) if current_iv >= max(historical_ivs) else \
           np.percentile(historical_ivs, 0) if current_iv <= min(historical_ivs) else \
           (sum(1 for iv in historical_ivs if iv <= current_iv) / len(historical_ivs)) * 100
    
    return round(rank, 2)


def get_implied_volatility(ticker_obj, period: str = "1y") -> tuple:
    """
    Obtiene la volatilidad implícita actual y el histórico
    
    Args:
        ticker_obj: Objeto yfinance ticker
        period: Período de histórico
    
    Returns:
        Tupla (current_iv, historical_ivs)
    """
    try:
        # Intentar obtener IV de las opciones
        options_dates = ticker_obj.options
        if not options_dates or len(options_dates) < 2:
            return None, []
        
        # Obtener la cadena de opciones más cercana (30-60 días)
        target_date = None
        today = datetime.now()
        
        for date_str in options_dates:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_to_exp = (exp_date - today).days
            if 25 <= days_to_exp <= 60:  # Buscar opciones ~30-45 DTE
                target_date = date_str
                break
        
        if not target_date:
            target_date = options_dates[0]  # Usar la más cercana
        
        # Obtener cadena de opciones
        opt_chain = ticker_obj.option_chain(target_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        if calls.empty or puts.empty:
            return None, []
        
        # Calcular IV promedio de ATM options
        current_price = ticker_obj.info.get('regularMarketPrice', ticker_obj.history(period='1d')['Close'].iloc[-1])
        
        # Filtrar opciones cerca del dinero (0.95 - 1.05 del precio actual)
        atm_calls = calls[(calls['strike'] >= current_price * 0.95) & 
                          (calls['strike'] <= current_price * 1.05)]
        atm_puts = puts[(puts['strike'] >= current_price * 0.95) & 
                        (puts['strike'] <= current_price * 1.05)]
        
        # Promediar IVs
        call_ivs = atm_calls['impliedVolatility'].dropna()
        put_ivs = atm_puts['impliedVolatility'].dropna()
        
        all_ivs = pd.concat([call_ivs, put_ivs])
        
        if all_ivs.empty:
            return None, []
        
        current_iv = all_ivs.mean() * 100  # Convertir a porcentaje
        
        # Obtener histórico de volatilidad realizada como proxy para IV histórica
        hist = ticker_obj.history(period=period)
        if hist.empty or len(hist) < 20:
            return current_iv, [current_iv]
        
        # Calcular volatilidad realizada histórica (proxy para IV)
        hist['returns'] = hist['Close'].pct_change()
        
        # Calcular volatilidad rolling de 20 días
        historical_vols = []
        for i in range(20, len(hist)):
            vol = hist['returns'].iloc[i-20:i].std() * np.sqrt(252) * 100
            if not np.isnan(vol):
                historical_vols.append(vol)
        
        # Ajustar las volatilidades históricas para que se alineen mejor con la IV actual
        # (La IV suele ser mayor que la volatilidad realizada)
        if historical_vols:
            avg_realized = np.mean(historical_vols)
            adjustment_factor = current_iv / avg_realized if avg_realized > 0 else 1.2
            historical_vols = [vol * adjustment_factor for vol in historical_vols]
        
        return current_iv, historical_vols
        
    except Exception as e:
        print(f"Error getting IV for symbol: {e}")
        return None, []


def calculate_risk_reversal(ticker_obj) -> tuple:
    """
    Calcula el Risk Reversal y Skew Rank
    
    Risk Reversal = IV(25Δ Call) - IV(25Δ Put)
    Skew Rank = Percentil del RR actual en el histórico
    
    Args:
        ticker_obj: Objeto yfinance ticker
    
    Returns:
        Tupla (risk_reversal, skew_rank)
    """
    try:
        options_dates = ticker_obj.options
        if not options_dates or len(options_dates) < 2:
            return 0.0, 50.0
        
        # Obtener la cadena de opciones
        target_date = None
        today = datetime.now()
        
        for date_str in options_dates:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_to_exp = (exp_date - today).days
            if 25 <= days_to_exp <= 60:
                target_date = date_str
                break
        
        if not target_date:
            target_date = options_dates[0]
        
        opt_chain = ticker_obj.option_chain(target_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        if calls.empty or puts.empty:
            return 0.0, 50.0
        
        # Obtener precio actual
        current_price = ticker_obj.info.get('regularMarketPrice', ticker_obj.history(period='1d')['Close'].iloc[-1])
        
        # Encontrar opciones OTM (25 delta aproximadamente = 10-15% OTM)
        otm_call_strike = current_price * 1.10
        otm_put_strike = current_price * 0.90
        
        # Encontrar las opciones más cercanas a estos strikes
        call_option = calls.iloc[(calls['strike'] - otm_call_strike).abs().argsort()[:1]]
        put_option = puts.iloc[(puts['strike'] - otm_put_strike).abs().argsort()[:1]]
        
        if call_option.empty or put_option.empty:
            return 0.0, 50.0
        
        call_iv = call_option['impliedVolatility'].iloc[0] * 100
        put_iv = put_option['impliedVolatility'].iloc[0] * 100
        
        # Risk Reversal = Call IV - Put IV
        risk_reversal = call_iv - put_iv
        
        # Para calcular Skew Rank, necesitamos histórico de RR
        # Simplificado: usamos la relación entre call y put IV
        # Positivo = Call skew (bullish), Negativo = Put skew (bearish)
        
        # Normalizar a escala 0-100
        # RR típicamente va de -10 a +10
        # Convertimos a percentil asumiendo distribución normal
        skew_rank = 50 + (risk_reversal / 0.2)  # Normalizar
        skew_rank = max(0, min(100, skew_rank))  # Limitar a 0-100
        
        return round(risk_reversal, 2), round(skew_rank, 2)
        
    except Exception as e:
        print(f"Error calculating risk reversal: {e}")
        return 0.0, 50.0


def fetch_symbol_data(symbol: str) -> Dict:
    """
    Obtiene todos los datos necesarios para un símbolo
    
    Args:
        symbol: Ticker symbol
    
    Returns:
        Diccionario con datos del símbolo o None si falla
    """
    print(f"Procesando {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Obtener IV y calcular IV Rank
        current_iv, historical_ivs = get_implied_volatility(ticker)
        
        if current_iv is None:
            print(f"  ⚠️  {symbol}: No se pudieron obtener datos de IV")
            return None
        
        iv_rank = calculate_iv_rank(current_iv, historical_ivs)
        
        # Calcular Risk Reversal y Skew Rank
        risk_reversal, skew_rank = calculate_risk_reversal(ticker)
        
        data = {
            'symbol': symbol,
            'iv_rank': iv_rank,
            'skew_rank': skew_rank,
            'current_iv': round(current_iv, 2),
            'risk_reversal': risk_reversal
        }
        
        print(f"  ✓ {symbol}: IV={current_iv:.1f}% (Rank={iv_rank:.1f}%), RR={risk_reversal:.2f}% (Skew={skew_rank:.1f}%)")
        
        return data
        
    except Exception as e:
        print(f"  ✗ {symbol}: Error - {e}")
        return None


def main():
    """
    Función principal que obtiene datos de todos los símbolos y genera el JSON
    """
    print("=" * 60)
    print("COMPASS DATA FETCHER")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Símbolos a procesar: {len(SYMBOLS)}")
    print("=" * 60)
    print()
    
    results = []
    
    for symbol in SYMBOLS:
        data = fetch_symbol_data(symbol)
        if data:
            results.append(data)
    
    print()
    print("=" * 60)
    print(f"Procesamiento completado: {len(results)}/{len(SYMBOLS)} símbolos")
    print("=" * 60)
    
    if not results:
        print("⚠️  ERROR: No se obtuvieron datos de ningún símbolo")
        return
    
    # Generar JSON
    output = {
        'last_updated': datetime.now().isoformat(),
        'symbols_count': len(results),
        'data': results
    }
    
    # Guardar archivo
    with open('compass-data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Archivo generado: compass-data.json")
    print(f"  - {len(results)} símbolos")
    print(f"  - Última actualización: {output['last_updated']}")


if __name__ == "__main__":
    main()
