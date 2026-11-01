"""Procesamiento diario de la CBA online (misma metodología que notebook_CBA.ipynb).

1. Lee todos los CSV de datos_crudos/.
2. Clasifica con Gemini solo los nombres nuevos (memoria en clasificaciones.csv).
3. Limpia, imputa, calcula la CBA, el índice de modelos emparejados y el test de Wilcoxon.
4. Guarda docs/datos.json, que es lo que lee la página web.
"""
import glob
import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

CONFIG = json.load(open("config.json", encoding="utf-8"))
CBA_INDEC = CONFIG["cba_indec"]
MES_INDEC = CONFIG["mes_indec"]
FAMILIA = 3.09
RUTA_CACHE = "clasificaciones.csv"

# ---------------------------------------------------------------- canasta
df_canasta = pd.read_csv("canasta_indec.csv")

# ---------------------------------------------------------------- precio por kilo
def extraer_kg(nombre, unidad):
    if unidad == "kg":
        return 1.0
    patron = r"(\d+(?:[.,]\d+)?)\s*(kg|kilos?|grs?|gramos|g|lts?|litros?|l|cc|ml)\b"
    coincidencia = re.search(patron, nombre.lower())
    if coincidencia is None:
        nombre_min = nombre.lower()
        if "huevo" in nombre_min:
            unidades = re.search(r"x\s*(\d+)|(\d+)\s*(?:unidades|uni|un|ud|u)\b", nombre_min)
            if unidades:
                cantidad = unidades.group(1) or unidades.group(2)
                return int(cantidad) * 0.05
            if "docena" in nombre_min:
                return 12 * 0.05
            if "maple" in nombre_min:
                return 30 * 0.05
        return None
    numero = float(coincidencia.group(1).replace(",", "."))
    medida = coincidencia.group(2)
    if medida in ["kg", "kilo", "kilos", "l", "lt", "lts", "litro", "litros"]:
        return numero
    return numero / 1000


# ---------------------------------------------------------------- clasificación con IA
ACLARACIONES = {
    "Jugos concentrados": "Acepta jugos en polvo para diluir y jugos concentrados para diluir en agua. Rechazá jugos listos para tomar.",
    "Soda": "Se refiere al agua con gas (Soda en sifón o botella). Rechazá maquinas, cilindros y accesorios.",
    "Paleta": "Se refiere al corte de carne vacuna. Rechaza la paleta cocida (fiambre).",
    "Caldos concentrados": "Se refiere a caldos en cubos o deshidratados para preparar. Rechaza sopas instantaneas, saborizantes y condimentos",
    "Tomate perita": "Se refiere al tomate perita fresco. Rechazá tomate en lata, cubeteado, triturado, en puré o en salsa.",
    "Lentejas": "Se refiere a lentejas secas. Rechazá lentejas en conserva, remojadas, en lata o preparadas.",
    "Arvejas": "Se refiere a arvejas secas. Rechazá arvejas en conserva, remojadas, en lata o preparadas.",
    "Yerba": "Se refiere a yerba mate común (tradicional, suave, con o sin palo). Rechazá yerbas compuestas, saborizadas o con hierbas.",
    "Aceite": "Se refiere a aceite de girasol, de maíz o mezcla. Rechazá aceite de oliva, aceites en aerosol o spray y aceites saborizados.",
    "Azúcar": "Se refiere a azúcar común blanca refinada (tipo A). Rechazá azúcar mascabo, rubia, orgánica, light, impalpable y edulcorantes.",
    "Leche": ("Solo leche fluida de vaca común: entera, descremada, liviana o larga vida. "
              "Rechazá leche chocolatada, saborizada, con cacao, en polvo, condensada, "
              "dulce de leche, bebidas vegetales y leches premium o funcionales "
              "(protein, con proteínas, barista, con vitaminas o fibras agregadas)."),
    "Pescado": ("Solo filet de merluza común, fresco o congelado, sin rebozar. "
                "Rechazá merluza negra (es otra especie), salmón, abadejo, langostinos, "
                "rebozados, empanados, medallones y alimento para mascotas."),
}
FRESCOS = ["Papa", "Batata", "Acelga", "Cebolla", "Lechuga", "Tomate perita", "Zanahoria", "Zapallo",
           "Manzana", "Mandarina", "Naranja", "Banana", "Pera"]
for producto in FRESCOS:
    ACLARACIONES.setdefault(producto, "Se refiere al producto fresco y entero. Rechazá versiones en lata, congeladas, "
                                      "ralladas, cubeteadas, deshidratadas, cocidas, en escamas o en puré.")

MODELO = "gemini-3.5-flash-lite"
cliente = None
if os.environ.get("GEMINI_API_KEY"):
    from google import genai
    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def consultar_ia(prompt, intentos=3):
    for intento in range(intentos):
        try:
            return cliente.models.generate_content(model=MODELO, contents=prompt).text
        except Exception:
            if intento == intentos - 1:
                raise
            espera = 10 * (intento + 1)
            print(f"  Servidor ocupado, reintento en {espera} segundos...")
            time.sleep(espera)


def clasificar_productos(producto_cba, rubro, nombres):
    lista = "\n".join(f"{i}. {n}" for i, n in enumerate(nombres, start=1))
    extra = ACLARACIONES.get(producto_cba, "")
    prompt = f"""Sos un asistente que ayuda a construir la Canasta Basica Alimentaria del INDEC (Argentina)

Producto de la canasta: "{producto_cba}",
Rubro de la canasta: "{rubro}"

Abajo hay una lista de productos encontrados en la web de un supermercado.
Indica cuales corresponden realmente a "{producto_cba}" como un alimento básico.

Criterios:
- Aceptá cualquier marca, tamaño o presentacion del mismo alimento.
- Rechazá productos que solo comparten el nombre o el sabor (por ejemplo, jugos o golosinas "sabor naranja" cuando se busca naranja).
- Rechazá productos elaborados a partir de ese alimento. combos y cosas que no son alimentos.

Lista:
{lista}
{extra}

Respondé SOLO con los números de los productos válidos separados por coma (por ejemplo: 1, 4, 7).
Si ninguno es válido, respondé: ninguno"""
    respuesta = consultar_ia(prompt)
    numeros = [int(x) for x in re.findall(r"\d+", respuesta)]
    return [i in numeros for i in range(1, len(nombres) + 1)]


archivos = sorted(glob.glob("datos_crudos/precios_*.csv"))
df_todo = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
df_todo["fecha"] = df_todo["fecha"].astype(str)

cache = pd.read_csv(RUTA_CACHE)
nuevos = []
for producto in df_canasta["producto"]:
    rubro = df_canasta.loc[df_canasta["producto"] == producto, "rubro"].iloc[0]
    ya = set(cache.loc[cache["producto_cba"] == producto, "nombre"])
    pendientes = [n for n in df_todo.loc[df_todo["producto_cba"] == producto, "nombre"].unique() if n not in ya]
    if not pendientes:
        continue
    if cliente is None:
        print(f"Sin clave de Gemini: '{producto}' queda con {len(pendientes)} nombres sin clasificar")
        continue
    try:
        respuestas = clasificar_productos(producto, rubro, pendientes)
    except Exception as error:
        print(f"ATENCIÓN: '{producto}' queda pendiente ({len(pendientes)} nombres): {error}")
        continue
    nuevos += [{"producto_cba": producto, "nombre": n, "valido": v} for n, v in zip(pendientes, respuestas)]
    print(f"{producto}: {sum(respuestas)} válidos de {len(pendientes)} nuevos")
    time.sleep(5)

if nuevos:
    cache = pd.concat([cache, pd.DataFrame(nuevos)], ignore_index=True)
    cache.to_csv(RUTA_CACHE, index=False)
print("Total en memoria:", len(cache))

# ---------------------------------------------------------------- panel
df_todo["kg"] = df_todo.apply(lambda f: extraer_kg(f["nombre"], f["unidad"]), axis=1)
df_todo["precio_kg"] = df_todo["precio_sin_desc"] / df_todo["kg"]
es_jugo = df_todo["producto_cba"] == "Jugos concentrados"
df_todo.loc[es_jugo, "precio_kg"] = df_todo.loc[es_jugo, "precio_sin_desc"] / (df_todo.loc[es_jugo, "kg"] * 1000 / 15)

memoria = cache.drop_duplicates(subset=["producto_cba", "nombre"])
df_todo = df_todo.merge(memoria, on=["producto_cba", "nombre"], how="left")

# ---------------------------------------------------------------- limpieza y CBA
df_limpio = df_todo[(df_todo["valido"] == True) & df_todo["precio_kg"].notna() & (df_todo["precio_sin_desc"] > 0)].copy()
es_fresco = df_limpio["producto_cba"].isin(FRESCOS)
granel = (df_limpio["unidad"] == "kg") | df_limpio["nombre"].str.contains(r"(?:x|por)\s*kg", case=False, regex=True)
hay_granel = granel.groupby([df_limpio["fecha"], df_limpio["cadena"], df_limpio["producto_cba"]]).transform("any")
df_limpio = df_limpio[~es_fresco | granel | ~hay_granel]
df_limpio = df_limpio[df_limpio["precio_kg"] >= 300]
mediana_prod = df_limpio.groupby(["fecha", "producto_cba"])["precio_kg"].transform("median")
df_limpio = df_limpio[(df_limpio["precio_kg"] >= mediana_prod / 3) & (df_limpio["precio_kg"] <= mediana_prod * 3)]

precios_prod = df_limpio.groupby(["fecha", "cadena", "producto_cba"])["precio_kg"].median().reset_index()

productos = df_canasta[["producto"]].drop_duplicates().rename(columns={"producto": "producto_cba"})
grilla_prod = precios_prod[["fecha", "cadena"]].drop_duplicates().merge(productos, how="cross")
prod_comp = grilla_prod.merge(precios_prod, on=["fecha", "cadena", "producto_cba"], how="left")
prod_comp["imputado"] = prod_comp["precio_kg"].isna()
prod_comp = prod_comp.sort_values(["cadena", "producto_cba", "fecha"])
prod_comp["precio_kg"] = prod_comp.groupby(["cadena", "producto_cba"])["precio_kg"].ffill(limit=3)
prod_comp["precio_kg"] = prod_comp["precio_kg"].fillna(
    prod_comp.groupby(["fecha", "producto_cba"])["precio_kg"].transform("mean"))
prod_comp = prod_comp.dropna(subset=["precio_kg"])

merge_canasta = df_canasta[["producto", "rubro", "cantidad_mensual"]]
precios_prod = precios_prod.merge(merge_canasta, left_on="producto_cba", right_on="producto")
prod_comp = prod_comp.merge(merge_canasta, left_on="producto_cba", right_on="producto")

precios_rubro = prod_comp.groupby(["fecha", "cadena", "rubro", "cantidad_mensual"])["precio_kg"].mean().reset_index()
cba = precios_rubro.groupby(["fecha", "cadena"]).size().reset_index()[["fecha", "cadena"]]

rubros_canasta = df_canasta[["rubro", "cantidad_mensual"]].drop_duplicates()
completo = cba.merge(rubros_canasta, how="cross").merge(
    precios_rubro[["fecha", "cadena", "rubro", "precio_kg"]], on=["fecha", "cadena", "rubro"], how="left")
completo = completo.sort_values(["cadena", "rubro", "fecha"])
completo["precio_kg"] = completo.groupby(["cadena", "rubro"])["precio_kg"].ffill(limit=3)
completo["precio_kg"] = completo["precio_kg"].fillna(completo.groupby(["fecha", "rubro"])["precio_kg"].transform("mean"))
completo["costo"] = completo["precio_kg"] * completo["cantidad_mensual"] / 1000
cba_final = completo.groupby(["fecha", "cadena"])["costo"].sum().reset_index(name="cba")

# ---------------------------------------------------------------- índice de modelos emparejados
base = df_limpio["fecha"].min()
p0 = (df_limpio[df_limpio["fecha"] == base]
      .groupby(["cadena", "producto_cba", "nombre"], as_index=False)["precio_kg"].mean()
      .rename(columns={"precio_kg": "p0"}))
pt = df_limpio.groupby(["fecha", "cadena", "producto_cba", "nombre"], as_index=False)["precio_kg"].mean()
emparejados = pt.merge(p0, on=["cadena", "producto_cba", "nombre"])
emparejados["relativo"] = emparejados["precio_kg"] / emparejados["p0"]
emparejados = emparejados[(emparejados["relativo"] >= 0.5) & (emparejados["relativo"] <= 2.0)].copy()
emparejados["log_rel"] = np.log(emparejados["relativo"])
ind_prod = emparejados.groupby(["fecha", "producto_cba"])["log_rel"].mean().apply(np.exp).reset_index(name="indice")
ind_prod = ind_prod.merge(df_canasta[["producto", "rubro"]], left_on="producto_cba", right_on="producto")
ind_rubro = ind_prod.groupby(["fecha", "rubro"])["indice"].mean().reset_index()

pesos = completo[completo["fecha"] == base].groupby("rubro")["costo"].mean()
pesos = pesos / pesos.sum()
ind_rubro["peso"] = ind_rubro["rubro"].map(pesos)
ind_rubro["ponderado"] = ind_rubro["indice"] * ind_rubro["peso"]
resumen = ind_rubro.groupby("fecha")[["ponderado", "peso"]].sum()
indice_cba = resumen["ponderado"] / resumen["peso"] * 100

cba_base = cba_final[cba_final["fecha"] == base]["cba"].mean()
serie = pd.DataFrame({"indice": indice_cba})
serie["cba_online"] = cba_base * serie["indice"] / 100

# ---------------------------------------------------------------- análisis
composicion = (pesos * 100).sort_values(ascending=False)
por_cadena = cba_final.groupby("cadena")["cba"].mean().sort_values()

prom = precios_prod.groupby(["producto_cba", "cadena"])["precio_kg"].mean().unstack()
wilcoxon = []
for a, b in [("Carrefour", "Jumbo"), ("Carrefour", "Dia"), ("Dia", "Jumbo")]:
    datos = prom[[a, b]].dropna()
    dif_log = np.log(datos[b] / datos[a])
    res = stats.wilcoxon(dif_log)
    wilcoxon.append({"cara": b, "barata": a, "n": int(len(datos)),
                     "dif_mediana": round(float((np.exp(dif_log.median()) - 1) * 100), 1),
                     "p": float(res.pvalue)})

# ---------------------------------------------------------------- salida para la página
ultimo = serie.index[-1]
cba_hoy = float(serie["cba_online"].iloc[-1])
NOMBRE = {"Dia": "Día"}
salida = {
    "actualizado": datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).strftime("%Y-%m-%d %H:%M"),
    "fecha_ultima": ultimo,
    "fecha_base": base,
    "dias": int(len(serie)),
    "precios_relevados": int(len(df_todo)),
    "precios_validos": int(len(df_limpio)),
    "indec": {"valor": CBA_INDEC, "mes": MES_INDEC},
    "hoy": {"cba": round(cba_hoy), "familia": round(cba_hoy * FAMILIA),
            "brecha": round((cba_hoy / CBA_INDEC - 1) * 100, 1),
            "var_acum": round(float(serie["indice"].iloc[-1] - 100), 2),
            "var_diaria": round(float(serie["indice"].pct_change().iloc[-1] * 100), 2) if len(serie) > 1 else None},
    "serie": [{"fecha": f, "indice": round(float(r["indice"]), 2), "cba": round(float(r["cba_online"]))}
              for f, r in serie.iterrows()],
    "cadenas": [{"cadena": NOMBRE.get(c, c), "cba": round(float(v)), "familia": round(float(v) * FAMILIA),
                 "vs_indec": round((float(v) / CBA_INDEC - 1) * 100, 1)} for c, v in por_cadena.items()],
    "composicion": [{"rubro": r, "pct": round(float(v), 1)} for r, v in composicion.items()],
    "wilcoxon": [{**w, "cara": NOMBRE.get(w["cara"], w["cara"]), "barata": NOMBRE.get(w["barata"], w["barata"])}
                 for w in wilcoxon],
    "imputados_pct": round(float(prod_comp["imputado"].mean() * 100), 1),
    "acierto_ia": "96–100%",
}
os.makedirs("docs", exist_ok=True)
json.dump(salida, open("docs/datos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
serie.round(2).to_csv("docs/serie_cba_online.csv", index_label="fecha")
print(f"Listo: {base} al {ultimo} ({len(serie)} días) | CBA online {cba_hoy:,.0f} | brecha {salida['hoy']['brecha']:+.1f}%")
