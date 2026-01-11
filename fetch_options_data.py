#!/usr/bin/env python3
"""
Compass Data Fetcher V2 - Pure Metrics Edition
Usa solo métricas reales: Risk Premium (IV - HV) y Risk Reversal
Sin aproximaciones ni invenciones
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Lista de símbolos a analizar
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WFC', 'DIS', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'TLT', 'COIN', 'PLTR', 'SOFI', 'UBER'
]

# Configuración
HV_PERIOD = 20  # Días para calcular volatilidad realizada


def calculate_realized_volatility(ticker_obj, period: int = 20) -> float:
    """
    Calcula la volatilidad realizada (HV) de los últimos N días
    
    Args:
        ticker_obj: Objeto yfinance ticker
        period: Días para calcular HV
    
    Returns:
        Volatilidad realizada anualizada en %
    """
    try:
        # Obtener histórico de precios
        hist = ticker_obj.history(period=f"{period + 10}d")
        
        if hist.empty or len(hist) < period:
            return None
        
        # Calcular returns
        returns = hist['Close'].pct_change().dropna()
        
        # Tomar últimos N días
        returns = returns.tail(period)
        
        if len(returns) < period:
            return None
        
        # Calcular volatilidad anualizada
        hv = returns.std() * np.sqrt(252) * 100
        
        return round(hv, 2)
        
    except Exception as e:
        print(f"  Error calculating HV: {e}")
        return None


def get_atm_implied_volatility(ticker_obj) -> Tuple[float, float]:
    """
    Obtiene la volatilidad implícita ATM promedio
    
    Args:
        ticker_obj: Objeto yfinance ticker
    
    Returns:
        Tupla (current_iv, current_price)
    """
    try:
        options_dates = ticker_obj.options
        if not options_dates or len(options_dates) < 2:
            return None, None
        
        # Buscar opciones entre 25-60 días
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
        
        # Obtener cadena de opciones
        opt_chain = ticker_obj.option_chain(target_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        if calls.empty or puts.empty:
            return None, None
        
        # Obtener precio actual
        try:
            current_price = ticker_obj.info.get('regularMarketPrice')
            if not current_price:
                current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
        except:
            current_price = ticker_obj.history(period='1d')['Close'].iloc[-1]
        
        # Filtrar opciones ATM (±5%)
        atm_calls = calls[
            (calls['strike'] >= current_price * 0.95) & 
            (calls['strike'] <= current_price * 1.05)
        ]
        atm_puts = puts[
            (puts['strike'] >= current_price * 0.95) & 
            (puts['strike'] <= current_price * 1.05)
        ]
        
        # Obtener IVs
        call_ivs = atm_calls['impliedVolatility'].dropna()
        put_ivs = atm_puts['impliedVolatility'].dropna()
        
        if call_ivs.empty and put_ivs.empty:
            return None, None
        
        # Promediar todas las IVs ATM
        all_ivs = pd.concat([call_ivs, put_ivs])
        current_iv = all_ivs.mean() * 100
        
        return round(current_iv, 2), current_price
        
    except Exception as e:
        print(f"  Error getting IV: {e}")
        return None, None


def calculate_risk_premium(current_iv: float, realized_vol: float) -> float:
    """
    Calcula el Risk Premium
    
    Risk Premium = IV - HV
    Positivo = Opciones caras (mercado paga prima)
    Negativo = Opciones baratas (no hay prima)
    
    Args:
        current_iv: Volatilidad implícita actual
        realized_vol: Volatilidad realizada
    
    Returns:
        Risk Premium en puntos porcentuales
    """
    if current_iv is None or realized_vol is None:
        return None
    
    return round(current_iv - realized_vol, 2)


def calculate_risk_reversal(ticker_obj, current_price: float) -> float:
    """
    Calcula el Risk Reversal real
    
    Risk Reversal = IV(25Δ Call) - IV(25Δ Put)
    Aproximamos 25Δ como ~10% OTM
    
    Args:
        ticker_obj: Objeto yfinance ticker
        current_price: Precio actual del subyacente
    
    Returns:
        Risk Reversal en puntos porcentuales
    """
    try:
        options_dates = ticker_obj.options
        if not options_dates:
            return None
        
        # Buscar fecha entre 25-60 días
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
            return None
        
        # Strikes OTM (~10%)
        otm_call_strike = current_price * 1.10
        otm_put_strike = current_price * 0.90
        
        # Encontrar opciones más cercanas
        call_option = calls.iloc[(calls['strike'] - otm_call_strike).abs().argsort()[:1]]
        put_option = puts.iloc[(puts['strike'] - otm_put_strike).abs().argsort()[:1]]
        
        if call_option.empty or put_option.empty:
            return None
        
        call_iv = call_option['impliedVolatility'].iloc[0] * 100
        put_iv = put_option['impliedVolatility'].iloc[0] * 100
        
        # Risk Reversal = Call IV - Put IV
        risk_reversal = call_iv - put_iv
        
        return round(risk_reversal, 2)
        
    except Exception as e:
        print(f"  Error calculating Risk Reversal: {e}")
        return None


def fetch_symbol_data(symbol: str) -> Dict:
    """
    Obtiene todos los datos necesarios para un símbolo
    
    Returns:
        Dict con: symbol, risk_premium, risk_reversal, current_iv, hv_20
    """
    print(f"Procesando {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Obtener IV actual
        current_iv, current_price = get_atm_implied_volatility(ticker)
        
        if current_iv is None or current_price is None:
            print(f"  ⚠️  {symbol}: No se pudo obtener IV")
            return None
        
        # 2. Calcular volatilidad realizada
        hv_20 = calculate_realized_volatility(ticker, HV_PERIOD)
        
        if hv_20 is None:
            print(f"  ⚠️  {symbol}: No se pudo calcular HV")
            return None
        
        # 3. Calcular Risk Premium
        risk_premium = calculate_risk_premium(current_iv, hv_20)
        
        # 4. Calcular Risk Reversal
        risk_reversal = calculate_risk_reversal(ticker, current_price)
        
        if risk_reversal is None:
            print(f"  ⚠️  {symbol}: No se pudo calcular Risk Reversal")
            return None
        
        data = {
            'symbol': symbol,
            'risk_premium': risk_premium,      # IV - HV (eje X)
            'risk_reversal': risk_reversal,    # Call IV - Put IV (eje Y)
            'current_iv': current_iv,
            'hv_20': hv_20,
            'iv_hv_ratio': round(current_iv / hv_20, 2) if hv_20 > 0 else None
        }
        
        print(f"  ✓ {symbol}:")
        print(f"    IV={current_iv:.1f}%, HV={hv_20:.1f}%")
        print(f"    Risk Premium={risk_premium:+.2f}% (IV-HV)")
        print(f"    Risk Reversal={risk_reversal:+.2f}% (C-P)")
        
        return data
        
    except Exception as e:
        print(f"  ✗ {symbol}: Error - {e}")
        return None


def main():
    """
    Función principal
    """
    print("=" * 60)
    print("COMPASS DATA FETCHER V2 - PURE METRICS")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Símbolos a procesar: {len(SYMBOLS)}")
    print()
    print("Métricas:")
    print("  • Risk Premium = IV - HV (opciones caras/baratas)")
    print("  • Risk Reversal = Call IV - Put IV (sesgo direccional)")
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
    
    # Estadísticas
    risk_premiums = [r['risk_premium'] for r in results if r['risk_premium'] is not None]
    risk_reversals = [r['risk_reversal'] for r in results if r['risk_reversal'] is not None]
    
    if risk_premiums:
        print()
        print("📊 Estadísticas del Mercado:")
        print(f"  Risk Premium promedio: {np.mean(risk_premiums):+.2f}%")
        print(f"  Risk Premium rango: {min(risk_premiums):+.2f}% a {max(risk_premiums):+.2f}%")
        print(f"  Risk Reversal promedio: {np.mean(risk_reversals):+.2f}%")
        print(f"  Risk Reversal rango: {min(risk_reversals):+.2f}% a {max(risk_reversals):+.2f}%")
    
    # Generar JSON
    output = {
        'last_updated': datetime.now().isoformat(),
        'symbols_count': len(results),
        'metrics': {
            'risk_premium': 'IV - Historical Volatility (20-day)',
            'risk_reversal': 'Call IV (25Δ) - Put IV (25Δ)',
            'interpretation': {
                'risk_premium_positive': 'Options expensive (market paying premium)',
                'risk_premium_negative': 'Options cheap (no premium)',
                'risk_reversal_positive': 'Call skew (bullish sentiment)',
                'risk_reversal_negative': 'Put skew (bearish sentiment)'
            }
        },
        'data': results
    }
    
    # Guardar archivo
    with open('compass-data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"✓ Archivo generado: compass-data.json")
    print(f"  - {len(results)} símbolos")
    print(f"  - Última actualización: {output['last_updated']}")
    print()
    print("🎯 Cuadrantes (nuevo sistema):")
    print("  Q1 (↗): Prima Alta + Call Skew → Movimiento alcista esperado y caro")
    print("  Q2 (↖): Prima Baja + Call Skew → Oportunidad alcista (opciones baratas)")
    print("  Q3 (↙): Prima Baja + Put Skew → Protección barata disponible")
    print("  Q4 (↘): Prima Alta + Put Skew → Pánico (protección muy cara)")


if __name__ == "__main__":
    main()
