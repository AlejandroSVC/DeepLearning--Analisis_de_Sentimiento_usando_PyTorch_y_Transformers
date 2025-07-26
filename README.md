# Análisis de Sentimiento usando Deep Learning, PyTorch y Hugging Face Transformers

Este script realiza un Análisis de Sentimientos sobre opiniones escritas en español, utilizando Deep Learning, PyTorch y la librería Hugging Face Transformers. El archivo de datos CSV debe contener opiniones en español, cada una en una fila. El flujo incluye: carga y preprocesamiento de datos, tokenización,  construcción de dataset y dataloader, carga de un modelo preentrenado (BERT), entrenamiento (opcional), evaluación y predicción de sentimientos. Los comentarios de cada sección explican detalladamente los pasos realizados.

## Importación de librerías necesarias

En esta sección importamos las librerías que utilizaremos, como pandas para manejo de datos, torch para Deep Learning, y las herramientas de Hugging Face para la gestión de modelos y tokenización.
```
import pandas as pd                                                              # Para manipulación de archivos CSV
import torch                                                                     # Para operaciones de deep learning
from torch.utils.data import Dataset, DataLoader                                 # Para crear datasets y loaders
from transformers import AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments                                                   # De Hugging Face
from sklearn.model_selection import train_test_split                             # Para dividir los datos
import numpy as np                                                               # Para operaciones numéricas
```
## Definición de parámetros y carga del archivo CSV

Aquí se definen los parámetros y se carga el archivo CSV con las opiniones.
```
csv_file = "opiniones.csv"                                                       # Nombre del archivo CSV con opiniones
text_column = "opinion"                                                          # Nombre de la columna que contiene los textos en español
label_column = "sentimiento"                                                     # Nombre de la columna con los sentimientos (0=negativo, 1=positivo)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"                  # ModeloBERT multilingüe para análisis de sentimientos
```

## Carga y preprocesamiento de datos

En esta sección se cargan los datos, y se preprocesan para eliminar valores nulos o duplicados.
```
df = pd.read_csv(csv_file)                                                       # Carga los datos del archivo CSV
df = df.dropna(subset=[text_column, label_column])                               # Elimina filas con valores nulos en texto o etiqueta
df = df.drop_duplicates(subset=[text_column])                                    # Elimina opiniones duplicadas
```

## División del dataset en entrenamiento y prueba

Aquí se divide el dataset en conjuntos de entrenamiento y prueba para evaluar el modelo.
```
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[text_column].tolist(),                                                    # Lista de textos para entrenamiento y validación
    df[label_column].tolist(),                                                   # Lista de etiquetas de sentimientos
    test_size=0.2,                                                               # Porcentaje del conjunto para validación
    random_state=42,                                                             # Semilla para reproducibilidad
    stratify=df[label_column].tolist()                                           # Estratificación por clase
)
```

## Tokenización de textos usando el modelo de Hugging Face

Se utiliza un tokenizador compatible con el modelo BERT multilingüe para convertir textos en tensores.
```
tokenizer = AutoTokenizer.from_pretrained(model_name)                            # Carga el tokenizador

def tokenize_texts(texts):                                                       # Función para tokenizar textos
    return tokenizer(
        texts,                                                                   # Lista de textos
        padding=True,                                                            # Relleno para igualar longitud
        truncation=True,                                                         # Truncar textos largos
        max_length=128,                                                          # Longitud máxima de tokens
        return_tensors="pt"                                                      # Salida como tensores de PyTorch
    )
```

## Creación de un Dataset personalizado para PyTorch

Aquí se define una clase dataset compatible con PyTorch y Hugging Face Trainer.
```
class OpinionesDataset(Dataset):                                                 # Dataset personalizado
    def __init__(self, texts, labels):                                           # Inicialización con textos y etiquetas
        encodings = tokenize_texts(texts)                                        # Tokenizar los textos
        # Convertir las etiquetas a cero-indexadas (de 1-5 a 0-4) para este modelo
        self.labels = torch.tensor([int(label)-1 for label in labels])           # Etiquetas como tensores, ajustadas
        self.encodings = encodings                                               # Almacena los tensores de entrada
        
    def __getitem__(self, idx):                                                  # Método para obtener un elemento
        item = {key: val[idx] for key, val in self.encodings.items()}            # Obtener los tensores de un índice
        item["labels"] = self.labels[idx]                                        # Añadir la etiqueta correspondiente
        return item                                                              # Retornar el diccionario

    def __len__(self):                                                           # Retornar la cantidad de ejemplos
        return len(self.labels)
```

## Creación de datasets y dataloaders

Se instancian los datasets de entrenamiento y validación.
```
train_dataset = OpinionesDataset(train_texts, train_labels)                      # Dataset de entrenamiento
val_dataset =   OpinionesDataset(val_texts, val_labels)                          # Dataset de validación
```

## Carga del modelo multilingüe preentrenado para clasificación de secuencias

Se utiliza un modelo BERT multilingüe con una capa de clasificación de sentimientos con 5 clases (1-5 estrellas).
```
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,                                                                  # Nombre del modelo preentrenado
    num_labels=5                                                                 # Número de clases de salida (1 a 5 estrellas)
)
```

## Definición de métricas de evaluación

Aquí se define una función para calcular la exactitud (accuracy) durante la evaluación.
```
def compute_metrics(eval_pred):                                                  # Función para métricas
    logits, labels = eval_pred                                                   # Salidas del modelo y etiquetas reales
    preds = np.argmax(logits, axis=1)                                            # Predicciones (clase con mayor probabilidad)
    accuracy = (preds == labels).mean()                                          # Cálculo de la exactitud
    return {"accuracy": accuracy}                                                # Retornar un diccionario con la métrica
```

## Configuración del entrenamiento

Se define los argumentos de entrenamiento, como número de épocas, batch size, y directorio de salida.
```
training_args = TrainingArguments(
    output_dir="./results",                                                      # Carpeta de resultados
    num_train_epochs=2,                                                          # Épocas de entrenamiento
    per_device_train_batch_size=8,                                               # Tamaño de batch para entrenamiento
    per_device_eval_batch_size=8,                                                # Tamaño de batch para validación
    evaluation_strategy="epoch",                                                 # Estrategia de evaluación
    save_strategy="epoch",                                                       # Guardado por época
    logging_dir="./logs",                                                        # Carpeta de logs
    logging_steps=20,                                                            # Pasos entre logs
    load_best_model_at_end=True,                                                 # Cargar el mejor modelo al final
    metric_for_best_model="accuracy"                                             # Métrica para seleccionar el mejor modelo
)
```

## Entrenamiento y evaluación del modelo

Se instancia el Trainer de Hugging Face, que gestiona el entrenamiento y la evaluación.
```
trainer = Trainer(
    model=model,                                                                 # El modelo BERT multilingüe preentrenado
    args=training_args,                                                          # Argumentos de entrenamiento
    train_dataset=train_dataset,                                                 # Dataset de entrenamiento
    eval_dataset=val_dataset,                                                    # Dataset de validación
    compute_metrics=compute_metrics                                              # Función de métricas
)
```
## Entrenar el modelo
```
trainer.train()                                                                  # Entrenamiento del modelo
```

## Evaluación del modelo sobre el conjunto de validación

Aquí se evalúa el modelo entrenado para obtener la exactitud en el conjunto de validación.
```
results = trainer.evaluate()                                                     # Evaluación
print("Resultados de evaluación:", results)                                      # Muestra las métricas
```

## Predicción de sentimientos en nuevas opiniones

Por último, se puede usar el modelo entrenado para predecir el sentimiento de nuevas opiniones.
```
def predecir_sentimiento(texto):                                                 # Función de predicción
    inputs = tokenize_texts([texto])                                             # Tokenizar el texto de entrada
    outputs = model(**inputs)                                                    # Pasar el texto por el modelo
    pred = torch.argmax(outputs.logits, dim=1).item() + 1                        # El modelo devuelve 0-4 (sumamos 1 para obtener 1-5)
    return f"{pred} estrellas"                                                   # Devolver la cantidad de estrellas
    ```
## Ejemplo de uso
```
opinion_nueva = "Este producto es excelente, me encantó."                        # Opinión de ejemplo
print("Sentimiento:", predecir_sentimiento(opinion_nueva))                       # Predicción de sentimiento
```
