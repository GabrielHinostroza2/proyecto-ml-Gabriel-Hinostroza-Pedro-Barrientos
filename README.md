# Proyecto ML Gabriel Hinostroza-Pedro Barrientos

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.0.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.
 # Proyecto ML – Airbnb Seattle (Kedro + Notebooks)

**Curso:** Machine Learning (MLY0100)  
**Nombres** Gabriel Hinostroza - Pedro Barrientos>  
**Video explicativo:** https://<link-a-tu-video>  ← *pegar aquí tu link*

---

## 🎯 Objetivo
Implementar un flujo completo de **ciencia de datos** usando el dataset **Airbnb Seattle** (Kaggle), desde:
1) comprensión y exploración,
2) preparación y **preprocesamiento**,
3) creación de **datasets listos para modelar** (regresión y clasificación),
4) con estructura reproducible usando **Kedro**.

---

## 📦 Estructura del proyecto (Kedro)
proyecto-ml-Gabriel Hinostroza-Pedro-Barrientos>/
├─ conf/
│ └─ base/
│ ├─ catalog.yml # Datasets (raw → primary → feature → model_input)
│ ├─ parameters.yml # Parámetros (test_size, random_state)
│ └─ logging.yml
├─ data/
│ ├─ 01_raw/ # CSV crudos (listings.csv, calendar.csv, reviews.csv)
│ ├─ 02_intermediate/ # Transformaciones intermedias
│ ├─ 03_primary/ # Datos limpios/canónicos
│ ├─ 04_feature/ # Matrices de features (X,y)
│ └─ 05_model_input/ # Splits de entrenamiento/prueba
├─ notebooks/
│ ├─ 01_business_understanding.ipynb
│ ├─ 02_data_understanding.ipynb
│ └─ 03_data_preparation.ipynb
├─ src/
│ └─ <paquete>/
│ ├─ pipeline_registry.py
│ └─ pipelines/
│ └─ data_processing/
│ ├─ nodes.py # limpieza, feature engineering, imputación, splits
│ └─ pipeline.py # 6 nodos encadenados
├─ pyproject.toml
└─ README.md

