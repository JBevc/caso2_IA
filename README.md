# Caso de Estudio 2 — Deep Learning: clasificación de razas de mascotas

Clasificador de las 35 razas de gatos y perros de la partición de Oxford-IIIT Pet entregada por la cátedra. Para cada
foto de test predice la raza y decide si se ofrece como sugerencia (`sugerir`) o se deriva a revisión humana
(`revisar`), según la confianza del modelo.

## Contenido

| Archivo / carpeta | Qué es |
| --- | --- |
| `caso2_deep_learning.ipynb` | Notebook con la solución completa. |
| `predicciones_test.csv` | Predicciones entregadas: `image_id,raza_predicha,decision`|
| `requirements.txt` | Dependencias mínimas |
| `data/` | Lugar donde van los paquetes de datos de la cátedra (no se incluyen en el repositorio) |
| `salidas/` | Resultados de la corrida entregada: partición, tabla de experimentos, métricas por raza, matriz de confusión, umbral de decisión, figuras y ficha del modelo final |

## Cómo ejecutar

1. Clonar el repositorio y copiar en `data/` los dos paquetes de la cátedra, sin descomprimir y con sus nombres
   originales:

   ```
   data/caso2_train.zip
   data/caso2_test.zip
   ```

2. Abrir `caso2_deep_learning.ipynb` en Google Colab con GPU (una T4 alcanza) o en Jupyter local con PyTorch y CUDA.
   En Colab, subir la carpeta `data/` junto al notebook o dejar los zips en una carpeta `caso2_IA` de *Mi unidad*
   de Google Drive: el notebook la monta y copia los datos. Si no encuentra los zips en ninguno de los dos lugares,
   intenta descargarlos de la carpeta compartida de la cátedra.
3. Ejecutar todo (*Entorno de ejecución → Ejecutar todo*). Las dependencias que falten se instalan solas.
4. Al terminar, las predicciones quedan en `predicciones_test.csv` (raíz) y en `salidas/predicciones_test.csv`.

## Datos y atribución

Oxford-IIIT Pet: O. M. Parkhi, A. Vedaldi, A. Zisserman y C. V. Jawahar, *Cats and Dogs*, IEEE CVPR 2012.
https://www.robots.ox.ac.uk/~vgg/data/pets/ — licencia CC BY-SA 4.0. Se usó únicamente la partición de 35 razas
entregada por la cátedra, sin imágenes ni etiquetas externas.