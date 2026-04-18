# UBA MIA — Aprendizaje por refuerzo I — Desafío práctico

Implementación de **Q-learning** sobre el entorno **FrozenLake-v1** (Gymnasium), con registro de recompensas por episodio, curva de convergencia y evaluación de la política greedy.

## Requisitos

- Python 3.10 o superior

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

Entrenamiento (por defecto: entorno resbaladizo, 20 000 episodios; guarda CSV y PNG en `informe/assets/`):

```powershell
python scripts/qlearning_frozenlake.py
```

Entorno determinístico (sin deslizamiento):

```powershell
python scripts/qlearning_frozenlake.py --no-slippery --tag deterministic
```

Evaluar la tabla Q guardada:

```powershell
python scripts/evaluate_policy.py --q-table informe/assets/qtable_slippery.npy
```

## Estructura

| Ruta | Descripción |
|------|-------------|
| `scripts/qlearning_frozenlake.py` | Entrenamiento Q-learning y gráficos |
| `scripts/evaluate_policy.py` | Tasa de éxito con política greedy |
| `informe/INFORME.md` | Cuerpo del informe (exportar a PDF) |
| `informe/assets/` | Figuras, CSV y `qtable_*.npy` generados por los scripts |

## Autor

Trabajo individual — Maestría en Inteligencia Artificial, UBA.
