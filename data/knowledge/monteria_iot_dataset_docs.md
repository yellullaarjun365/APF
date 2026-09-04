# Montería, Colombia IoT Water-Quality Datasets -- Documentation Summary

Paraphrased from the published documentation of the Mendeley Data IoT monitoring datasets by Baena-Navarro, Carriazo-Regino, Torres-Hoyos and colleagues, which were referenced during the design of this project's synthetic data generator. This file describes the real-world datasets themselves (what they measured and how), not general tilapia biology.

## What the datasets are

A series of related datasets published on Mendeley Data document IoT-based water-quality and fish-health monitoring carried out on Nile tilapia aquaculture ponds in Montería, Colombia, during 2024. Data collection ran from January to June 2024 (six months), using IoT sensors placed directly in the ponds, with readings taken at either hourly or 6-hourly intervals depending on the dataset version.

## What was measured

Across the dataset family, the consistently recorded parameters are:
- **Water temperature (°C)**
- **Dissolved oxygen, DO (mg/L)**
- **pH**
- **Turbidity (NTU)** -- a measure of water clarity, which affects light penetration and can indicate algae levels or suspended solids
- **Average fish weight (g)** and **survival rate (%)**, as fish-health indicators tracked alongside the water-quality readings
- Some versions of the dataset also record **disease occurrence (case counts)**, **oxygenation interventions (yes/no)**, and **corrective interventions taken**, directly linking specific water-quality events to the actions farm operators took in response.

## Derived risk indicators included in the data

The dataset documentation describes several pre-computed risk flags built directly into the data, useful as a model for what "at a glance" risk signals look like in practice:
- **Thermal Risk Index**: flagged "High" or "Normal" based on temperature readings.
- **Low Oxygen Alert**: flagged "Critical" if DO fell below 5 mg/L, otherwise "Safe."
- **Health Status**: an overall "At Risk" or "Stable" flag for the fish, derived from combining the thermal and oxygen alerts together, rather than from either factor alone.

## Purpose and intended use

The dataset's stated purpose is to support predictive modeling for water-quality management and fish-health outcomes in aquaculture, including training machine-learning models (the associated research used Random Forest and Support Vector Machine approaches) to predict water-quality risk and reduce fish mortality in real-world tropical pond conditions with limited technical infrastructure -- a similar goal to what this project's own forecasting model is built for, though using different specific inputs and outputs.

## Practical relevance to this project

This dataset is real evidence that a DO threshold around 5 mg/L is used in practice as an actionable "critical" cutoff by other tilapia monitoring systems, consistent with the DO guidance already in this knowledge base. It also demonstrates that combining multiple risk factors (temperature + oxygen, rather than either alone) into a single health status is a standard and useful practice, not an oversimplification.
