# Robótica Inteligente

## **Cómo ejecutar la práctica 2**

### **Entrenamiento Evolutivo**
1. **Configura el entorno**  
   Instala dependencias (revisar `requirements.txt` si existe/necesario).
   ```
   pip install gymnasium neat-python robobopy robobosim stable-baselines3
   ```

2. **Entrena con NEAT**  
   Desde la carpeta de la versión:
   ```
   cd p2/v3/
   python train.py
   ```

   Esto:
   - Inicializa la población
   - Ejecuta el ciclo evolutivo
   - Guarda el genoma ganador (`models_neat/winner_genome.pkl`)
   - Salva estadísticas y backups

3. **Valida el resultado**
   ```
   python validate.py 
   ```
   Esto ejecuta el genoma puro y guarda los trayectos y recompensas.

### **Modo Híbrido (AE+AR, Práctica 2.3)**
Si tienes un modelo de AR entrenado (por ejemplo, de la práctica 1), puedes combinar políticas:
   ```
   python validate.py --policy_type ae_ar --ar_model_path /ruta/al/modelo/SAC.zip
   ```
El script alterna entre AE y AR según la observación del entorno.

4. **Generación de gráficas**
    ```
    python generate_visual.py
    ```
Genera las gráficas del entrenamiento, incluida la trayectoria si se ejecuto la validación.


---

## **Detalles del entorno (`env.py`)**

- **Observaciones:**  
  12 dimensiones que incluyen posición del agente, posición de objetivo, blob visible, posición y tamaño del blob rojo, y sensado IR para obstáculos.

- **Acciones:**  
  Dos valores continuos para las ruedas [-1, 1].

- **Función de *fitness*:**  
  - Progreso hacia el cilindro
  - Mantener el objetivo visible y centrado
  - Penalización fuerte por colisión/proximidad
  - Recompensa elevada al alcanzar el objetivo

---

## **Gráficas y análisiso**

- **Gráfica de aprendizaje** (fitness por generación)
- **Estructura del ganador NEAT** (`winner_genome`)
- **Especies evolucionadas** (si aplica)
- **Recorridos 2D** del robot por episodio (`trajectory`)
- **Comparativa de episodios logrados vs. total**
