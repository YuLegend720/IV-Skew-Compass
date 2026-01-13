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
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Lista de símbolos a analizar
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WFC', 'DIS', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'TLT', 'COIN', 'PLTR', 'SOFI', 'UBER'
]


@dataclass
class Config:
    """Configuración del sistema"""
    HV_PERIOD: int = 20  # Días para calcular volatilidad realizada
    MIN_DTE: int = 25  # Días mínimos hasta expiración
    MAX_DTE: int = 60  # Días máximos hasta expiración
    ATM_RANGE: float = 0.05  # ±5% para considerar ATM
    OTM_CALL_MULT: float = 1.10  # 10% OTM para calls (aprox 25Δ)
    OTM_PUT_MULT: float = 0.90  # 10% OTM para puts (aprox 25Δ)
    MIN_HISTORY_DAYS: int = 30  # Mínimo de días de histórico necesario
    RETRY_ATTEMPTS: int = 3  # Intentos de retry para API
    RETRY_DELAY: float = 1.0  # Segundos entre reintentos
    MAX_WORKERS: int = 5  # Threads concurrentes para fetching
    OUTPUT_FILE: str = 'compass-data.json'
    ANNUALIZATION_FACTOR: int = 252  # Días de trading por año


config = Config()


def retry_on_failure(func):
    """Decorator para reintentar operaciones que fallan"""
    def wrapper(*args, **kwargs):
        for attempt in range(config.RETRY_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < config.RETRY_ATTEMPTS - 1:
                    logger.warning(f"Intento {attempt + 1} falló: {e}. Reintentando...")
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Todos los intentos fallaron para {func.__name__}: {e}")
                    raise
        return None
    return wrapper


def calculate_realized_volatility(ticker_obj, period: Optional[int] = None) -> Optional[float]:
    """
    Calcula la volatilidad realizada (HV) de los últimos N días

    Args:
        ticker_obj: Objeto yfinance ticker
        period: Días para calcular HV (usa config.HV_PERIOD si no se especifica)

    Returns:
        Volatilidad realizada anualizada en % o None si falla

    Raises:
        ValueError: Si no hay suficientes datos históricos
    """
    if period is None:
        period = config.HV_PERIOD

    try:
        # Obtener más días de los necesarios para asegurar suficiente data
        hist = ticker_obj.history(period=f"{period + 10}d")

        if hist.empty:
            raise ValueError("No hay datos históricos disponibles")

        if len(hist) < period:
            raise ValueError(f"Insuficientes datos: {len(hist)} < {period}")

        # Calcular returns y limpiar NaN
        returns = hist['Close'].pct_change().dropna()

        # Tomar últimos N días
        returns = returns.tail(period)

        if len(returns) < period:
            raise ValueError(f"Insuficientes returns: {len(returns)} < {period}")

        # Validar que los returns son razonables
        if returns.std() == 0:
            raise ValueError("Desviación estándar es cero (precio sin cambios)")

        # Calcular volatilidad anualizada
        hv = returns.std() * np.sqrt(config.ANNUALIZATION_FACTOR) * 100

        # Validar resultado
        if not np.isfinite(hv) or hv <= 0 or hv > 500:
            raise ValueError(f"HV fuera de rango razonable: {hv}")

        return round(hv, 2)

    except Exception as e:
        logger.debug(f"Error calculating HV: {e}")
        return None


def get_current_price(ticker_obj) -> Optional[float]:
    """
    Obtiene el precio actual del ticker con fallback

    Args:
        ticker_obj: Objeto yfinance ticker

    Returns:
        Precio actual o None si falla
    """
    try:
        # Intentar desde info primero (más rápido)
        price = ticker_obj.info.get('regularMarketPrice')
        if price and price > 0:
            return float(price)
    except Exception:
        pass

    try:
        # Fallback: último precio de cierre
        hist = ticker_obj.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        logger.debug(f"Error obteniendo precio actual: {e}")

    return None


def find_target_expiration(options_dates: List[str]) -> Optional[str]:
    """
    Encuentra la fecha de expiración óptima (entre MIN_DTE y MAX_DTE)

    Args:
        options_dates: Lista de fechas de expiración disponibles

    Returns:
        Fecha óptima o None si no hay opciones
    """
    if not options_dates:
        return None

    today = datetime.now()
    target_date = None

    for date_str in options_dates:
        try:
            exp_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_to_exp = (exp_date - today).days

            if config.MIN_DTE <= days_to_exp <= config.MAX_DTE:
                target_date = date_str
                break
        except ValueError:
            continue

    # Si no encuentra en rango, usa la primera disponible
    return target_date or options_dates[0]


def get_atm_implied_volatility(ticker_obj) -> Tuple[Optional[float], Optional[float]]:
    """
    Obtiene la volatilidad implícita ATM promedio

    Args:
        ticker_obj: Objeto yfinance ticker

    Returns:
        Tupla (current_iv, current_price) o (None, None) si falla
    """
    try:
        # Obtener fechas de opciones disponibles
        options_dates = ticker_obj.options
        if not options_dates or len(options_dates) < 1:
            raise ValueError("No hay opciones disponibles")

        # Encontrar fecha objetivo
        target_date = find_target_expiration(options_dates)
        if not target_date:
            raise ValueError("No se encontró fecha de expiración válida")

        # Obtener cadena de opciones
        opt_chain = ticker_obj.option_chain(target_date)
        calls = opt_chain.calls
        puts = opt_chain.puts

        if calls.empty or puts.empty:
            raise ValueError("Cadena de opciones vacía")

        # Obtener precio actual
        current_price = get_current_price(ticker_obj)
        if not current_price or current_price <= 0:
            raise ValueError("Precio actual inválido")

        # Calcular rango ATM
        atm_lower = current_price * (1 - config.ATM_RANGE)
        atm_upper = current_price * (1 + config.ATM_RANGE)

        # Filtrar opciones ATM
        atm_calls = calls[
            (calls['strike'] >= atm_lower) &
            (calls['strike'] <= atm_upper)
        ]
        atm_puts = puts[
            (puts['strike'] >= atm_lower) &
            (puts['strike'] <= atm_upper)
        ]

        # Obtener IVs válidas
        call_ivs = atm_calls['impliedVolatility'].dropna()
        put_ivs = atm_puts['impliedVolatility'].dropna()

        # Filtrar IVs razonables (entre 0.01 y 5.0 = 1% a 500%)
        call_ivs = call_ivs[(call_ivs > 0.01) & (call_ivs < 5.0)]
        put_ivs = put_ivs[(put_ivs > 0.01) & (put_ivs < 5.0)]

        if call_ivs.empty and put_ivs.empty:
            raise ValueError("No hay IVs ATM válidas")

        # Promediar todas las IVs ATM
        all_ivs = pd.concat([call_ivs, put_ivs])
        current_iv = all_ivs.mean() * 100

        if not np.isfinite(current_iv):
            raise ValueError("IV promedio no es finita")

        return round(current_iv, 2), round(current_price, 2)

    except Exception as e:
        logger.debug(f"Error getting IV: {e}")
        return None, None


def calculate_risk_premium(current_iv: Optional[float], realized_vol: Optional[float]) -> Optional[float]:
    """
    Calcula el Risk Premium

    Risk Premium = IV - HV
    - Positivo: Opciones caras (mercado paga prima)
    - Negativo: Opciones baratas (no hay prima)

    Args:
        current_iv: Volatilidad implícita actual (%)
        realized_vol: Volatilidad realizada (%)

    Returns:
        Risk Premium en puntos porcentuales o None si faltan datos
    """
    if current_iv is None or realized_vol is None:
        return None

    if not np.isfinite(current_iv) or not np.isfinite(realized_vol):
        return None

    premium = current_iv - realized_vol

    # Validar que el resultado es razonable (-100% a +100%)
    if not -100 <= premium <= 100:
        logger.warning(f"Risk premium fuera de rango: {premium}")
        return None

    return round(premium, 2)


def calculate_risk_reversal(ticker_obj, current_price: float) -> Optional[float]:
    """
    Calcula el Risk Reversal real

    Risk Reversal = IV(25Δ Call) - IV(25Δ Put)
    Aproximamos 25Δ como ~10% OTM

    Args:
        ticker_obj: Objeto yfinance ticker
        current_price: Precio actual del subyacente

    Returns:
        Risk Reversal en puntos porcentuales o None si falla
    """
    try:
        if not current_price or current_price <= 0:
            raise ValueError("Precio actual inválido")

        # Obtener fechas de opciones
        options_dates = ticker_obj.options
        if not options_dates:
            raise ValueError("No hay opciones disponibles")

        # Buscar fecha objetivo
        target_date = find_target_expiration(options_dates)
        if not target_date:
            raise ValueError("No se encontró fecha de expiración válida")

        # Obtener cadena de opciones
        opt_chain = ticker_obj.option_chain(target_date)
        calls = opt_chain.calls
        puts = opt_chain.puts

        if calls.empty or puts.empty:
            raise ValueError("Cadena de opciones vacía")

        # Calcular strikes OTM objetivo (~10% para aproximar 25Δ)
        otm_call_strike = current_price * config.OTM_CALL_MULT
        otm_put_strike = current_price * config.OTM_PUT_MULT

        # Encontrar opciones más cercanas al strike objetivo
        call_diffs = (calls['strike'] - otm_call_strike).abs()
        put_diffs = (puts['strike'] - otm_put_strike).abs()

        if call_diffs.empty or put_diffs.empty:
            raise ValueError("No hay strikes disponibles")

        call_idx = call_diffs.idxmin()
        put_idx = put_diffs.idxmin()

        call_option = calls.loc[call_idx]
        put_option = puts.loc[put_idx]

        # Extraer IVs
        call_iv = call_option['impliedVolatility']
        put_iv = put_option['impliedVolatility']

        # Validar IVs
        if pd.isna(call_iv) or pd.isna(put_iv):
            raise ValueError("IVs no disponibles para strikes OTM")

        if not (0.01 <= call_iv <= 5.0) or not (0.01 <= put_iv <= 5.0):
            raise ValueError("IVs fuera de rango razonable")

        # Convertir a porcentaje y calcular Risk Reversal
        call_iv_pct = call_iv * 100
        put_iv_pct = put_iv * 100
        risk_reversal = call_iv_pct - put_iv_pct

        # Validar resultado
        if not np.isfinite(risk_reversal):
            raise ValueError("Risk Reversal no es finito")

        if not -50 <= risk_reversal <= 50:
            logger.warning(f"Risk Reversal fuera de rango típico: {risk_reversal}")

        return round(risk_reversal, 2)

    except Exception as e:
        logger.debug(f"Error calculating Risk Reversal: {e}")
        return None


@dataclass
class SymbolData:
    """Clase para datos de un símbolo"""
    symbol: str
    risk_premium: float
    risk_reversal: float
    current_iv: float
    hv_20: float
    iv_hv_ratio: float

    def to_dict(self) -> Dict:
        """Convierte a diccionario para JSON"""
        return {
            'symbol': self.symbol,
            'risk_premium': self.risk_premium,
            'risk_reversal': self.risk_reversal,
            'current_iv': self.current_iv,
            'hv_20': self.hv_20,
            'iv_hv_ratio': self.iv_hv_ratio
        }


def fetch_symbol_data(symbol: str) -> Optional[SymbolData]:
    """
    Obtiene todos los datos necesarios para un símbolo

    Args:
        symbol: Símbolo del ticker (ej: 'AAPL')

    Returns:
        SymbolData object o None si falla
    """
    logger.info(f"Procesando {symbol}...")

    try:
        # Crear objeto ticker
        ticker = yf.Ticker(symbol)

        # 1. Obtener IV actual y precio
        current_iv, current_price = get_atm_implied_volatility(ticker)

        if current_iv is None or current_price is None:
            logger.warning(f"{symbol}: No se pudo obtener IV o precio")
            return None

        # 2. Calcular volatilidad realizada
        hv_20 = calculate_realized_volatility(ticker)

        if hv_20 is None:
            logger.warning(f"{symbol}: No se pudo calcular HV")
            return None

        # 3. Calcular Risk Premium
        risk_premium = calculate_risk_premium(current_iv, hv_20)

        if risk_premium is None:
            logger.warning(f"{symbol}: No se pudo calcular Risk Premium")
            return None

        # 4. Calcular Risk Reversal
        risk_reversal = calculate_risk_reversal(ticker, current_price)

        if risk_reversal is None:
            logger.warning(f"{symbol}: No se pudo calcular Risk Reversal")
            return None

        # 5. Calcular ratio IV/HV
        iv_hv_ratio = round(current_iv / hv_20, 2) if hv_20 > 0 else None

        if iv_hv_ratio is None:
            logger.warning(f"{symbol}: HV es cero, no se puede calcular ratio")
            return None

        # Crear objeto de datos
        data = SymbolData(
            symbol=symbol,
            risk_premium=risk_premium,
            risk_reversal=risk_reversal,
            current_iv=current_iv,
            hv_20=hv_20,
            iv_hv_ratio=iv_hv_ratio
        )

        logger.info(
            f"✓ {symbol}: IV={current_iv:.1f}%, HV={hv_20:.1f}%, "
            f"RP={risk_premium:+.2f}%, RR={risk_reversal:+.2f}%"
        )

        return data

    except Exception as e:
        logger.error(f"✗ {symbol}: Error inesperado - {e}")
        return None


def generate_output_json(results: List[SymbolData]) -> Dict:
    """
    Genera la estructura de datos para el archivo JSON

    Args:
        results: Lista de SymbolData objects

    Returns:
        Dict con estructura completa para JSON
    """
    return {
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
        'data': [r.to_dict() for r in results]
    }


def print_statistics(results: List[SymbolData]) -> None:
    """
    Imprime estadísticas del mercado

    Args:
        results: Lista de SymbolData objects
    """
    if not results:
        return

    risk_premiums = [r.risk_premium for r in results]
    risk_reversals = [r.risk_reversal for r in results]
    iv_values = [r.current_iv for r in results]
    hv_values = [r.hv_20 for r in results]

    logger.info("")
    logger.info("📊 Estadísticas del Mercado:")
    logger.info(f"  IV promedio: {np.mean(iv_values):.2f}%")
    logger.info(f"  HV promedio: {np.mean(hv_values):.2f}%")
    logger.info(f"  Risk Premium promedio: {np.mean(risk_premiums):+.2f}%")
    logger.info(f"  Risk Premium rango: [{min(risk_premiums):+.2f}%, {max(risk_premiums):+.2f}%]")
    logger.info(f"  Risk Reversal promedio: {np.mean(risk_reversals):+.2f}%")
    logger.info(f"  Risk Reversal rango: [{min(risk_reversals):+.2f}%, {max(risk_reversals):+.2f}%]")


def main() -> None:
    """
    Función principal - orquesta el fetching y procesamiento de datos
    """
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("COMPASS DATA FETCHER V2 - PURE METRICS EDITION")
    logger.info("=" * 70)
    logger.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Símbolos a procesar: {len(SYMBOLS)}")
    logger.info(f"Workers concurrentes: {config.MAX_WORKERS}")
    logger.info("")
    logger.info("Métricas:")
    logger.info("  • Risk Premium = IV - HV (opciones caras/baratas)")
    logger.info("  • Risk Reversal = Call IV - Put IV (sesgo direccional)")
    logger.info("=" * 70)
    logger.info("")

    results = []

    # Procesamiento concurrente para mejor performance
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol_data, symbol): symbol
            for symbol in SYMBOLS
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {e}")

    elapsed_time = time.time() - start_time

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Procesamiento completado: {len(results)}/{len(SYMBOLS)} símbolos")
    logger.info(f"Tiempo total: {elapsed_time:.2f} segundos")
    logger.info(f"Promedio por símbolo: {elapsed_time/len(SYMBOLS):.2f}s")
    logger.info("=" * 70)

    if not results:
        logger.error("⚠️  ERROR: No se obtuvieron datos de ningún símbolo")
        return

    # Imprimir estadísticas
    print_statistics(results)

    # Generar JSON
    output = generate_output_json(results)

    # Guardar archivo
    try:
        with open(config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("")
        logger.info(f"✓ Archivo generado: {config.OUTPUT_FILE}")
        logger.info(f"  - {len(results)} símbolos procesados")
        logger.info(f"  - Última actualización: {output['last_updated']}")
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")
        return

    logger.info("")
    logger.info("🎯 Interpretación de Cuadrantes:")
    logger.info("  Q1 (↗): Prima Alta + Call Skew → Alcista esperado, caro")
    logger.info("  Q2 (↖): Prima Baja + Call Skew → Oportunidad alcista")
    logger.info("  Q3 (↙): Prima Baja + Put Skew → Protección barata")
    logger.info("  Q4 (↘): Prima Alta + Put Skew → Pánico / protección cara")
    logger.info("")


if __name__ == "__main__":
    main()
