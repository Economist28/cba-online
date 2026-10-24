"""Recolección diaria de precios para la CBA online (Carrefour, Jumbo, Día - VTEX).
Lo ejecuta GitHub Actions todos los días y guarda datos_crudos/precios_YYYY-MM-DD.csv
"""
import os
import sys
import time
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

CADENAS = {
    "Carrefour": "https://www.carrefour.com.ar",
    "Jumbo": "https://www.jumbo.com.ar",
    "Dia": "https://diaonline.supermercadosdia.com.ar",
}

TERMINOS = {"Tomate envasado": "tomate triturado", "Queso de rallar": "queso rallado",
            "Jugos concentrados": "jugo en polvo", "Queso cuartirolo": "cuartirolo",
            "Carnaza común": "carnaza", "Pan": "pan frances",
            "Pescado": "filet de merluza", "Caldos concentrados": "caldo"}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def buscar_productos(cadena, texto, cantidad=10, intentos=3):
    url = (f"{CADENAS[cadena]}/api/catalog_system/pub/products/search/"
           f"{quote(texto)}?_from=0&_to={cantidad - 1}")
    for intento in range(intentos):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code in (200, 206):
                break
            print(f"  {cadena} / {texto}: status {r.status_code}")
        except requests.RequestException as error:
            print(f"  {cadena} / {texto}: {error}")
        time.sleep(5 * (intento + 1))
    else:
        return []

    filas = []
    for p in r.json():
        try:
            item = p["items"][0]
            oferta = item["sellers"][0]["commertialOffer"]
        except (KeyError, IndexError):
            continue
        filas.append({
            "cadena": cadena,
            "busqueda": texto,
            "nombre": p.get("productName"),
            "precio_lista": oferta.get("ListPrice"),
            "precio_oferta": oferta.get("Price"),
            "precio_sin_desc": oferta.get("PriceWithoutDiscount"),
            "unidad": item.get("measurementUnit"),
            "multiplicador": item.get("unitMultiplier"),
        })
    return filas


def main():
    hoy = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).date().isoformat()
    os.makedirs("datos_crudos", exist_ok=True)
    ruta = f"datos_crudos/precios_{hoy}.csv"
    if os.path.exists(ruta):
        print(f"Ya existe {ruta}, no se vuelve a recolectar.")
        return

    df_canasta = pd.read_csv("canasta_indec.csv")
    filas = []
    for producto in df_canasta["producto"]:
        texto = TERMINOS.get(producto, producto)
        for cadena in CADENAS:
            resultados = buscar_productos(cadena, texto, cantidad=10)
            for fila in resultados:
                fila["fecha"] = hoy
                fila["producto_cba"] = producto
            filas.extend(resultados)
            time.sleep(1)

    df_crudo = pd.DataFrame(filas)
    if df_crudo.empty:
        sys.exit("ERROR: no se obtuvo ningún precio (¿bloqueo de las cadenas?).")

    df_crudo.to_csv(ruta, index=False)
    print(f"Guardado {ruta}: {df_crudo.shape}")
    conteo = df_crudo["cadena"].value_counts()
    print(conteo.to_string())
    for cadena in CADENAS:
        if conteo.get(cadena, 0) == 0:
            print(f"ATENCIÓN: {cadena} no devolvió precios hoy.")


if __name__ == "__main__":
    main()
