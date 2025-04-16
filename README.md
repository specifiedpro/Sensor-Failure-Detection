# Autoencoder-based Detector for Distinguishing Process Anomaly and Sensor Failure

**Chia-Yen Lee, Kai Chang, & Chien Ho**  
Pages 7130–7145 | Received 08 Jul 2023, Accepted 04 Feb 2024, Published online: 21 Feb 2024  

Cite this article:  
[https://doi.org/10.1080/00207543.2024.2318794](https://doi.org/10.1080/00207543.2024.2318794)

---

## Description

This repository contains the code for the paper **“Autoencoder-based Detector for Distinguishing Process Anomaly and Sensor Failure”** by Chia-Yen Lee, Kai Chang, and Chien Ho. The study proposes a framework that uses autoencoder-based control limits to dynamically and accurately detect when an observed anomaly is due to the manufacturing process itself, or instead caused by a sensor failure. It is validated via numerical simulations and a real-world semiconductor assembly use case.

### Abstract

> Anomaly detection is a frequently discussed topic in manufacturing. However, the issues of anomaly detection are typically attributed to the manufacturing process or equipment itself. In practice, the sensor responsible for collecting data and monitoring values may fail, leading to a biased detection result – false alarm. In such cases, replacing the sensor is necessary instead of performing equipment maintenance. This study proposes an effective framework embedded with autoencoder-based control limits that can dynamically distinguish sensor anomaly from process anomaly in real-time. We conduct a simulation numerical study and an empirical study of semiconductor assembling manufacturers to validate the proposed framework. The results show that the proposed model outperforms other benchmark methods and can successfully identify sensor failures, even under conditions of (1) large variations in process values or sensor values and (2) heteroscedasticity effect. This is particularly beneficial in various practical applications where sensors are used for numerical measurements and support equipment maintenance.

**Keywords**: Prognostics and health management, sensor failure, anomaly detection, deep learning, autoencoder

---

## Citation

> **Chia-Yen Lee, Kai Chang, & Chien Ho**  
> “Autoencoder-based Detector for Distinguishing Process Anomaly and Sensor Failure.”  
> *International Journal of Production Research*, Pages 7130–7145.  
> DOI: [10.1080/00207543.2024.2318794](https://doi.org/10.1080/00207543.2024.2318794)

No potential conflict of interest was reported by the authors.

Data supporting the findings of this study are available within the article and/or its supplementary materials.

---

## Repository Overview

This repository is organized into:

- **`src/`**  
  Contains the core source code for data generation, shaping/windowing, model definitions, threshold computations, and demo scripts.

- **`experiments/`**  
  Houses different experiment scenarios, including scripts for numerical studies or real-world tests with various anomaly conditions.

- **`tests/`**  
  Contains simple unit tests for data generation, models, and shaping.

Check out the `README.md` inside each folder or inline docstrings for further instructions on usage.

---

## Access & Journals

Taylor & Francis Online is offering access to the latest two full volumes of Engineering & Technology journals for free for 14 days. Sign in [here](https://www.tandfonline.com/) to start your access.

For more details, please see the [Taylor & Francis Online](https://www.tandfonline.com/) website.

---

## License

Please see the [LICENSE](LICENSE) file for details on how this repository’s content is licensed.
