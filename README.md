# Caso de estudio 2 - Deep Learning: clasificación de razas de mascotas

Inteligencia Artificial I (UCU) · Equipo: Bevc, Olivera, Tellechea · Docentes: Juan Francisco Kurucz, Bruno Cattaneo
Entrega: 16/11/2026 · Defensa: 23/11/2026

Clasificador de 35 razas de gatos y perros (Oxford-IIIT Pet) para sugerir la raza en las fichas de un portal de
adopción, con una decisión `sugerir` / `revisar` por foto según la confianza del modelo.

## Estructura del repositorio

```
caso2_ia/
├── README.md                     # este archivo: cómo ejecutar y qué hay en cada carpeta
├── requirements.txt              # dependencias mínimas (las exactas de cada corrida: salidas/requirements.txt)
├── .gitignore                    # excluye datos, pesos y archivos temporales
├── caso2_deep_learning.ipynb     # notebook completo, corre de principio a fin
├── predicciones_test.csv         # ENTREGA: image_id,raza_predicha,decision (copia de salidas/)
├── informe/
│   ├── informe_caso_2.pdf        # informe final
│   └── figuras/                  # figuras usadas en el informe (copiadas de salidas/figuras/)
├── data/                         # NO se versiona (ver .gitignore)
│   ├── caso2_train.zip           # paquete de entrenamiento entregado por la cátedra
│   └── caso2_test.zip            # paquete de test entregado por la cátedra
└── salidas/                      # generado por el notebook; se versiona todo salvo los pesos
    ├── requirements.txt          # versiones exactas de la corrida reportada
    ├── versiones.json
    ├── split_train_val.csv       # partición train/validación (80/20 estratificada, semilla 42)
    ├── eda_resumen.json
    ├── experimentos.csv          # tabla de pruebas: modelo, preparación, métricas, tiempo
    ├── modelo_final_meta.json    # arquitectura, pesos preentrenados, clases, normalización, umbral
    ├── modelo_final.pt           # NO se versiona (>100 MB); ver "Modelo final"
    ├── metricas_por_raza.csv
    ├── confusiones_top.csv
    ├── umbrales.csv
    ├── criterio.json             # umbral sugerir/revisar y su desempeño en validación
    ├── predicciones_test.csv
    └── figuras/                  # fig01 ... fig11
```

Qué sí y qué no va al repositorio:

| Va al repo | No va (está en `.gitignore`) | Por qué |
| --- | --- | --- |
| Notebook, README, requirements, informe PDF, figuras | `data/` (zips e imágenes descomprimidas) | 320 MB en total; GitHub limita a 100 MB por archivo y la cátedra ya distribuye los datos |
| Todas las salidas en CSV/JSON y `predicciones_test.csv` | `salidas/modelo_final.pt` | ~110 MB; el notebook lo reconstruye y la ficha `modelo_final_meta.json` queda versionada |
| | `.ipynb_checkpoints/`, `__pycache__/` | temporales |

Si quieren versionar los pesos igual, usar [Git LFS](https://git-lfs.com) (`git lfs track "salidas/*.pt"`) o subirlos como
*asset* de un *release* del repositorio y poner el enlace acá: **[COMPLETAR: enlace a los pesos, si se publican]**.

## Cómo ejecutar

1. Datos. El notebook los busca en este orden y usa el primero que encuentra:
   - `data/` junto al notebook: los zips (`caso2_train.zip`, `caso2_test.zip`) o las carpetas descomprimidas
     (`paquete_train_alumnos/`, `paquete_test_alumnos/`). Es lo normal al correr en local.
   - En Colab, una carpeta `caso2_IA` en *Mi unidad* de Google Drive con los mismos zips o carpetas
     (configurable en `DRIVE_DATA_DIR`, celda 1). El notebook pide permiso para montar Drive y copia los datos.
     Es la opción recomendada en Colab.
   - Descarga pública de la carpeta compartida de la cátedra (último recurso; Drive a veces limita descargas de
     archivos grandes).
2. Abrir `caso2_deep_learning.ipynb` en Google Colab (entorno de ejecución con GPU; una T4 alcanza) o en Jupyter
   local con PyTorch y CUDA. En Colab: subir el notebook y la carpeta `data/`, o montar Drive y ajustar solo las
   dos rutas `TRAIN_ZIP` y `TEST_ZIP` de la celda **1. Configuración** (es el único lugar con rutas).
3. *Entorno de ejecución → Ejecutar todo*. No hay que editar código.
4. Al terminar, `salidas/predicciones_test.csv` (y la copia en la raíz) es el archivo de entrega.

Tiempo estimado en una T4 con los cinco experimentos: 60 a 90 minutos. Para comprobar que todo corre, poner
`QUICK_TEST = True` en la celda 1 (10 % de los datos, 1 época por experimento, ~5 minutos); esa corrida **no**
sirve para la entrega.

## Dependencias

Python 3.10+, `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pillow`
(`requirements.txt`). En Colab ya vienen instaladas. Las versiones exactas con las que se obtuvieron los
resultados del informe quedan en `salidas/requirements.txt` y `salidas/versiones.json`.

## Reproducibilidad

- Semilla 42 en Python, NumPy, PyTorch y CUDA; cuDNN en modo determinista; `DataLoader` con generador y
  `worker_init_fn` fijos.
- La partición train/validación está en `salidas/split_train_val.csv` y es la misma para los cinco experimentos.
- El conjunto de test solo se lee en la celda **13**, después de fijar el modelo final y el umbral con validación.
- El modelo final se elige automáticamente por F1 macro en validación (celda 9); la regla está escrita antes de
  correr los experimentos.

## Modelo final

Pesos en `salidas/modelo_final.pt` (no versionado) y ficha en `salidas/modelo_final_meta.json`, que indica la
arquitectura y la fuente y versión de los pesos preentrenados (`torchvision.models`:
`ConvNeXt_Tiny_Weights.IMAGENET1K_V1` o `ResNet50_Weights.IMAGENET1K_V2`, según el experimento ganador).
Para reconstruirlo basta con volver a ejecutar el notebook (semilla fija) o, con los pesos a mano:

```python
import json, torch
from torchvision import models
meta = json.load(open("salidas/modelo_final_meta.json"))
m = models.convnext_tiny()                                   # o models.resnet50(), según meta["arquitectura"]
m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(meta["clases"]))  # m.fc en ResNet50
m.load_state_dict(torch.load("salidas/modelo_final.pt", map_location="cpu")); m.eval()
```

Preparación de entrada: lado menor a 256, recorte central 224×224, normalización con media y desvío de ImageNet
(`meta["normalizacion"]`). Criterio de decisión: `sugerir` si la probabilidad máxima ≥ `meta["configuracion"]`
/ `salidas/criterio.json → umbral`; `revisar` en caso contrario.

## Datos y atribución

Oxford-IIIT Pet: O. M. Parkhi, A. Vedaldi, A. Zisserman y C. V. Jawahar, *Cats and Dogs*, CVPR 2012.
https://www.robots.ox.ac.uk/~vgg/data/pets/ — licencia CC BY-SA 4.0. Se usa únicamente la partición de 35 razas
preparada por la cátedra; no se incorporaron imágenes ni etiquetas por fuera de esa entrega. El código de este
repositorio se publica bajo la misma licencia CC BY-SA 4.0 en lo que deriva del dataset (figuras con fotos).

## Declaración de uso de IA

Ver sección 8.3 del informe (`informe/informe_caso_2.pdf`).
