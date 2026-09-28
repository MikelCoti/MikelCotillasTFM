# Retail Inventory Risk con Geometric Deep Learning

Proyecto de **Data Science aplicado a gestión de inventario** desarrollado como parte de un Trabajo de Fin de Máster.

El objetivo es analizar si el uso de **Geometric Deep Learning (GDL)** puede mejorar la previsión de demanda y ayudar a tomar mejores decisiones de reposición de inventario.

## Descripción

La aplicación utiliza datos del dataset **Walmart M5** para generar previsiones de demanda a 7 días por producto y tienda.

Se comparan distintos enfoques:

- **LightGBM** como modelo base.
- Red neuronal sin conexiones entre productos.
- **GraphSAGE** utilizando relaciones entre el mismo producto en distintas tiendas.
- GraphSAGE utilizando similitud entre productos.
- Un modelo GraphSAGE que combina ambos tipos de relaciones.

Las previsiones se utilizan posteriormente en una **simulación de inventario** para analizar métricas como:

- Nivel de servicio / fill rate.
- Demanda no satisfecha.
- Días con rotura de stock.
- Gasto de reposición.
- Inventario medio.
- Contribución financiera simulada.

## Dashboard

El proyecto incluye una aplicación desarrollada con **Streamlit** con tres áreas principales:

### Executive Summary

Permite identificar productos con mayor riesgo estimado de rotura de stock, consultar previsiones de demanda y realizar simulaciones sencillas de reposición.

El riesgo se calcula utilizando una distribución binomial negativa cuya dispersión se calibra con los errores históricos de las previsiones.

### Inventory Simulation

Permite comparar los distintos modelos de forecasting bajo diferentes restricciones de presupuesto de compras.

### Business Analysis

Permite analizar los resultados:

- Por categoría: **FOODS, HOUSEHOLD y HOBBIES**.
- Por producto y tienda.
- Bajo diferentes presupuestos, lead times y niveles de stock de seguridad.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- LightGBM
- PyTorch
- PyTorch Geometric
- SciPy
- Streamlit

## Estructura principal

```text
├── dashboard.py
├── pages/
│   ├── 1_Inventory_Simulation.py
│   └── 2_Business_Analysis.py
├── src/
│   ├── prepare_data.py
│   ├── build_graph.py
│   ├── train_model.py
│   ├── train_graphsage.py
│   ├── run_graph_ablation.py
│   ├── simulate_inventory.py
│   ├── analyze_inventory.py
│   └── calibrate_demand_distribution.py
├── tests/
└── outputs/
```

## Ejecución

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Preparar los datos y entrenar los modelos:

```bash
python src/prepare_data.py
python src/build_graph.py
python src/train_model.py
python src/train_graphsage.py
```

Calibrar la distribución utilizada para estimar el riesgo:

```bash
python src/calibrate_demand_distribution.py
```

Ejecutar el dashboard:

```bash
streamlit run dashboard.py
```

## Dataset

El proyecto utiliza el dataset público **M5 Forecasting - Accuracy**, basado en datos históricos de ventas de Walmart.

Los archivos originales del dataset no se incluyen en este repositorio debido a su tamaño.

## Limitaciones

Este proyecto es un **MVP de apoyo a decisiones**, no un sistema operativo de gestión de inventario.

El dataset M5 contiene ventas históricas, pero no proporciona inventario real ni etiquetas fiables de roturas de stock. Por este motivo:

- El inventario utilizado en la aplicación es simulado.
- Los costes y presupuestos son supuestos experimentales.
- Las métricas financieras son estimaciones simuladas.
- Las probabilidades de riesgo representan incertidumbre sobre la demanda observada, no probabilidades verificadas de rotura de stock real.

## Conclusión

El proyecto muestra cómo combinar **forecasting, redes neuronales sobre grafos y simulación de inventario** para evaluar no solo la precisión predictiva de un modelo, sino también su posible impacto sobre decisiones de negocio.

Uno de los principales resultados del proyecto es que **una previsión más precisa no implica necesariamente una mejor decisión de inventario**, lo que demuestra la importancia de evaluar los modelos de Data Science también desde una perspectiva operativa.
