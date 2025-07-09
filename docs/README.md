[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Desarrollar un agente de IA para jugar Pong mediante código de C++.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `Pong Masters`
* **Integrantes**:

  * Galindo Quevedo Mauricio Jesus – 20241420 (Documentación y fundamentación teórica)
  * Lopez Rojas, Yaritza Milagros – 202410328 (Desarrollo de la arquitectura)
  * Ocampo Lecca, Mariela del Carmen – 202410258 (Pruebas y benchmarking)
  * Quesada Polar Alejandro Martín – 202410209 (Implementación del modelo)
  * Sánchez Arce, Mariano Andrés – 202410653 (Documentación y demo)

---


### Requisitos e instalación

1. **Compilador**: GCC 11 o superior

2. **Dependencias**:

  * CMake 3.14+ (recomendado 3.18+)
  * `<random>`
  * `<algorithm>`
  * `<ctime>`
  * `<memory>`
  * `<cmath>`
  * `<vector>`
  * `<string>`
  * `<iostream>`
  * `<array>`
  * `<numeric>`
  * `<stdexcept>`
  * `<fstream>`
  * `<unordered_map>`

3. **Instalación**:

   ```bash
   # Clonar el repositorio (ajusta según la ruta real)
   git clone https://github.com/TU-USUARIO/pongmasters.git
   ```

---
### 1. Investigación teórica

**Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.

#### 1.1 Historia y evolución – MLP como aproximador universal

Sosa Jerez y Zamora Alvarado (2022) describen la evolución del perceptrón simple hacia el perceptrón multicapa (MLP), apoyándose en el Teorema de Aproximación Universal, que asegura que un MLP con al menos una capa oculta puede aproximar cualquier función continua, lo cual es esencial para su uso como aproximador en tareas de regresión y resolución de ecuaciones diferenciales [1](#ref1).

#### 1.2 Arquitectura – CNN y LSTM

- **CNN**: La tesis de Gálvez Siuce (2023) presenta en detalle cómo diseñar y entrenar arquitecturas CNN, describiendo sus componentes principales (capas convolucionales, pooling, fully connected) y aplicándolas a tareas de clasificación con alta precisión (~98 %) [3](#ref3).

- **LSTM**: La investigación de la Universidad Pontificia Comillas (2023) muestra cómo las redes LSTM manejan dependencias temporales en series de tiempo (p. ej., precios de Bitcoin), comparándolas favorablemente con modelos tradicionales como ARIMA [4](#ref4).

#### 1.3 Aprendizaje y optimizadores

- **Descenso de gradiente (SGD)**:  
  Richaud (2021) explica claramente el *Descenso de Gradiente Estocástico (SGD)*, incluyendo su fórmula básica:

  $$
  w_{k+1} = w_k - \alpha \nabla C(w_k)
  $$

  donde:

- \( w_k \): vector de pesos en el paso \( k \)
- \( \alpha \): tasa de aprendizaje
- \( \nabla C(w_k) \): gradiente del costo respecto a los pesos

  SGD es un método simple y eficiente, especialmente útil cuando se trabaja con grandes cantidades de datos. Entre sus ventajas se encuentra su capacidad de converger rápidamente en entornos ruidosos. Sin embargo, puede presentar oscilaciones y una alta sensibilidad al tamaño del batch si no se ajustan adecuadamente sus hiperparámetros [6](#ref6).


- **Optimizador Adam**:  
  El mismo autor también cubre el optimizador *Adam*, que ajusta dinámicamente la tasa de aprendizaje mediante momentos del gradiente:

  - \( m_t \): estimación del **primer momento** (media del gradiente),
  - \( v_t \): estimación del **segundo momento** (varianza del gradiente),
  - Correcciones de sesgo para \( \hat{m}_t \) y \( \hat{v}_t \) (momentos ajustados).

  Fórmula de actualización:

  $$
  \begin{aligned}
  m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla C(w_t) \\\\
  v_t &= \beta_2 v_{t-1} + (1 - \beta_2)(\nabla C(w_t))^2 \\\\
  \hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad ; \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\\\
  w_{t+1} &= w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  \end{aligned}
  $$

  Adam combina lo mejor de dos mundos: la adaptabilidad de AdaGrad y la estabilidad de RMSProp. Se destaca por su rendimiento robusto en tareas con muchos parámetros y gradientes ruidosos [6](#ref6).

---

### 2. Diseño e implementación

| Archivo             | Patrón de diseño               | Descripción breve                                                  |
|----------------------|--------------------------------|----------------------------------------------------------------------|
| `nn_activation.h`    | Strategy, Polimorfismo        | Activaciones intercambiables (`ReLU`, `Sigmoid`).                   |
| `nn_optimizer.h`     | Strategy, Polimorfismo        | Estrategias de optimización intercambiables (`SGD`, `Adam`).        |
| `nn_loss.h`          | Strategy, Polimorfismo        | Funciones de pérdida intercambiables (`MSELoss`, `BCELoss`).        |
| `nn_interfaces.h`    | Interfaz, Polimorfismo        | Base común para capas, optimizadores y pérdidas.                    |
| Todo el proyecto     | Factory Method (potencial)    | Selección flexible de activación u optimizador en configuración.    |


#### 2.1 Patrón **Strategy**

**Descripción:**  
Permite intercambiar algoritmos de forma flexible sin modificar el cliente, encapsulando cada estrategia en una clase separada.

**Dónde se usa:**
- En `nn_activation.h`:
  - Las clases `ReLU` y `Sigmoid` implementan **estrategias de activación intercambiables**.
  - Ambas heredan de `ILayer`, permitiendo alternarlas según la necesidad.
- En `nn_optimizer.h`:
  - `SGD` y `Adam` implementan estrategias de optimización intercambiables.
  - Ambas implementan `IOptimizer` para utilizarlas sin cambiar el cliente.
- En `nn_loss.h`:
  - `MSELoss` y `BCELoss` implementan estrategias de función de pérdida intercambiables.

#### 2.2 Patrón **Factory Method (ligero)**

**Descripción:**  
Permite que subclases decidan qué objeto concreto instanciar, entregando flexibilidad en la creación de objetos.

**Dónde se usa:**  
Si bien no se implementa explícitamente, la estructura de interfaces y sus implementaciones (`ILayer` -> `ReLU`, `Sigmoid`; `IOptimizer` -> `SGD`, `Adam`) **permite utilizar Factory Method en la construcción de la red, seleccionando la activación y optimizador de forma flexible en tiempo de inicialización.**


#### 2.3 Patrón **Template Method (implícito)**

**Descripción:**  
Define el esqueleto de un algoritmo, permitiendo a las subclases redefinir pasos específicos.

**Dónde se usa:**
- Las interfaces `ILayer`, `IOptimizer`, `ILoss` definen operaciones como `forward`, `backward`, `update`, `loss`, `loss_gradient` que son implementadas en cada subclase, permitiendo reutilizar estructura y extender comportamientos de forma controlada.

#### 2.4 Patrón **Polimorfismo e Interfaz**

**Descripción:**  
Permite que múltiples clases compartan un mismo tipo de interfaz para utilizarse de forma intercambiable.

**Dónde se usa:**
- Las interfaces `ILayer`, `IOptimizer`, `ILoss` permiten que cualquier función de entrenamiento utilice activaciones, optimizadores y funciones de pérdida de forma intercambiable, manteniendo flexibilidad y extensibilidad en el sistema.

#### 2.5 **Estructura de carpetas**

  ```
  pongmasters-draft/
  ├── cmake-build-debug/
  ├── docs/
  │   └── README.md
  ├── include/
  │   └── utec/
  │       ├── agent/
  │       │   ├── EnvGym.h
  │       │   ├── PongAgent.h
  │       │   └── State.h
  │       ├── algebra/
  │       │   └── tensor.h
  │       └── nn/
  │           ├── activation.h
  │           ├── dense.h
  │           ├── interfaces.h
  │           ├── loss.h
  │           ├── neural_network.h
  │           └── optimizer.h
  ├── src/
  │   └── utec/
  │       └── agent/
  │           ├── EnvGym.cpp
  │           ├── PongAgent.cpp
  │           └── train_agent.cpp
  └── tests/
      ├── test_agent_env.cpp
      ├── test_neural_network.cpp
      └── test_tensor.cpp
  ```

#### 2.6 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

#### 4.1 Métricas globales (promedio sobre experimentos analizados):
- **Iteraciones:** 100-500 épocas según configuración.
- **Tiempo total de entrenamiento:** entre 5s y 137s dependiendo de epochs, max_steps y episodios.
- **Error final promedio:** 0.0007 - 0.08 según dataset y configuración.
- **Reducción de error inicial a final:** típicamente entre 90-99% en configuraciones con suficientes epochs y datos.

Estos datos se pueden observar en nuestro excel: [Analisis_Variables_Eficiencia_Entrenamiento_Pong.xlsx](https://docs.google.com/spreadsheets/d/1NrIKIr0SgqGlAIu140GT6Db6w9tRGxkbH1DCnRCKGzg/edit?usp=sharing)

#### 4.2 Análisis del efecto de cada variable en el rendimiento - Proyecto Pong

##### 4.2.1 Epochs (Épocas)
- **Aumentar epochs:**
  - Permite refinar pesos y **reducir el error final** si el loss sigue bajando.
  - **Incrementa el tiempo de entrenamiento**, especialmente con datasets grandes.
  - Mejora la precisión final hasta estabilizarse si se usa un learning rate adecuado.

- **Disminuir epochs:**
  - **Reduce el tiempo de entrenamiento.**
  - Puede causar **subentrenamiento (underfitting)** con error final alto y baja precisión.

##### 4.2.2 Batch Size
- **Aumentar batch size:**
  - Menos actualizaciones de pesos por epoch, entrenando más rápido.
  - Puede reducir la generalización si es demasiado grande.
  - Útil para aprovechar paralelización y reducir tiempos en GPU/CPU.

- **Disminuir batch size:**
  - Más actualizaciones por epoch, mejor capacidad de generalización.
  - Entrenamiento más lento pero con mejor convergencia en muchos casos.

##### 4.2.3 Max Steps
- **Aumentar max_steps:**
  - Más datos por episodio, mayor diversidad de estados.
  - Mejora representación de datos y **reduce error final**.
  - Aumenta significativamente el tiempo de recolección.

  - **Disminuir max_steps:**
    - Reduce tiempo de recolección.
    - Menos diversidad de datos, posible underfitting.

##### 4.2.4 Episodes
- **Aumentar episodes:**
  - Genera más datos para entrenamiento, mejorando generalización.
  - Reduce error inicial promedio.
  - Incrementa tiempo de recolección antes del entrenamiento.

- **Disminuir episodes:**
  - Reduce tiempo de recolección.
  - Genera datasets pequeños, limitando aprendizaje.

##### 4.2.5 Resumen general
| Variable   | Aumentar →                                  | Disminuir →                            |
|------------|---------------------------------------------|----------------------------------------|
| **Epochs**     | ↓ Error final, ↑ tiempo                     | ↑ Error final, ↓ tiempo               |
| **Batch Size** | ↓ tiempo por epoch, posible ↓ generalización | ↑ generalización, ↑ tiempo            |
| **Max Steps**  | ↑ diversidad de datos, ↓ error final        | ↓ diversidad, posible underfitting    |
| **Episodes**   | ↑ datos, ↓ error inicial                   | ↓ datos, ↑ error inicial              |

#### 4.3 Ventajas/Desventajas del flujo actual:

**Ventajas:**
- Código ligero, dependencias controladas (uso de C++ y librerías propias).
- Permite control preciso de variables como epochs, batch size, max_steps y episodios para análisis fino.
- Capacidad de evaluar el impacto de cada variable de forma aislada en la reducción de error y eficiencia del entrenamiento.

**Desventajas:**
- **Sin paralelización:** El entrenamiento es secuencial, lo que genera tiempos de entrenamiento prolongados en configuraciones de 500 epochs o más (hasta 137s).
- **Uso de CPU sin BLAS:** Multiplicaciones de matrices y forward/backward pass no están optimizadas a nivel bajo, afectando el rendimiento en datasets grandes.
- La recolección de datos con max_steps y episodios altos genera tiempos prolongados antes del entrenamiento.

---

### 5. Trabajo en equipo

| Tarea                     | Miembro           | Rol                       |
| ------------------------- |-------------------| ------------------------- |
| Investigación teórica     | Galindo Mauricio  | Documentar bases teóricas |
| Diseño de la arquitectura | Lopez Yaritza     | UML y esquemas de clases  |
| Implementación del modelo | Quesada Alejandro | Código C++ de la NN       |
| Pruebas y benchmarking    | Ocampo Mariela    | Generación de métricas    |
| Documentación y demo      | Mariano Sanchez   | Tutorial y video demo     |

---

### 6. Conclusiones

* **Logros**: Se logró implementar desde cero una red neuronal funcional en C++ con control total de las capas, funciones de activación, funciones de pérdida y optimizadores, integrándola con un entorno Pong propio para generación de datasets en entornos de refuerzo. Se consiguió entrenar el modelo con distintos hiperparámetros (`epochs`, `batch size`, `max_steps`, `episodes`) y medir con precisión su impacto en el error inicial, error final, tiempo de entrenamiento y capacidad de generalización. Asimismo, se generaron tablas y análisis comparativos que fortalecen el entendimiento experimental sobre el comportamiento de redes neuronales en C++.

* **Evaluación**: El sistema desarrollado demostró ser robusto para propósitos de investigación académica, mostrando reducciones consistentes de error inicial a error final de hasta un 99% en configuraciones óptimas, y permitiendo entrenamientos controlados entre 5 y 137 segundos según el tamaño del dataset y el número de épocas. Además, la arquitectura de código utilizada, con patrones de diseño como `Strategy` e `Interface`, permitió la modularidad y el fácil intercambio de funciones de activación, pérdida y optimización, asegurando un flujo de entrenamiento flexible y escalable.

* **Aprendizajes**: A nivel técnico, se consolidó la comprensión práctica de `backpropagation`, cálculo de gradientes y manejo de estructuras de datos en C++ orientadas a redes neuronales. Se comprendió el rol crítico de los hiperparámetros:
  - **`Epochs`**: Aumentar épocas reduce error final pero incrementa el tiempo de entrenamiento, especialmente útil si el loss continúa disminuyendo.
  - **`Batch Size`**: Tamaños grandes reducen el tiempo de entrenamiento por epoch pero pueden disminuir la capacidad de generalización, mientras que tamaños pequeños permiten una convergencia más suave.
  - **`Max Steps` y `Episodes`**: Impactan directamente en la diversidad y cantidad de datos recolectados, afectando el error inicial y final y permitiendo evaluar la robustez de la red en entornos de refuerzo.

  Se comprendió también el impacto que cada variable tiene en el pipeline de entrenamiento, permitiendo diseñar experimentos controlados, analizar resultados con un enfoque científico y comprender cuándo la limitación del modelo depende de falta de datos o de la arquitectura/hyperparametrización.

* **Recomendaciones**: Para fases posteriores de investigación y optimización:
  - **Escalar datasets** con más episodios y pasos máximos para robustecer la generalización del modelo en entornos complejos.
  - **Implementar BLAS (OpenBLAS/Intel MKL)** para acelerar multiplicaciones de matrices durante forward y backward pass, reduciendo tiempos de entrenamiento y permitiendo experimentar con redes más profundas sin penalización de tiempo excesiva.
  - **Paralelizar el entrenamiento por lotes** utilizando OpenMP o CUDA para mejorar la eficiencia en entrenamientos con batch sizes grandes.
  - **Explorar planificadores de learning rate dinámicos**, que reduzcan automáticamente la tasa de aprendizaje en caso de estancamiento, mejorando la convergencia sin necesidad de reiniciar manualmente.
  - Experimentar con **otras arquitecturas como CNN o LSTM** para extender el entrenamiento a problemas más complejos, como visión por computador o series de tiempo en entornos de refuerzo.
  - Mantener el enfoque modular de la arquitectura, para facilitar la integración de nuevos experimentos y el cambio de configuraciones de forma ágil.

En conclusión, el desarrollo de este proyecto ha permitido comprender profundamente el proceso de diseño, entrenamiento y análisis de redes neuronales implementadas en C++ desde cero, fortaleciendo competencias de investigación, programación y optimización, sentando una base sólida para escalar a proyectos más ambiciosos en el campo de inteligencia artificial y aprendizaje profundo.

---

### 7. Bibliografía

> <a name="ref1">[1]</a> L. V. Sosa Jerez and L. C. Zamora Alvarado, *Estructura de redes neuronales (MLP) y su aplicación como aproximador universal*, Monografía de grado, Universidad Distrital, 2022. [En línea]. Disponible en: https://repository.udistrital.edu.co/server/api/core/bitstreams/3373a7c7-b053-467a-aea5-2914b289c666/content

> <a name="ref2">[2]</a> A. Autor, *Memoria de investigación sobre redes neuronales convolucionales*, Tesis, Universidad de Salamanca, 2022. [En línea]. Disponible en: https://gredos.usal.es/bitstream/handle/10366/163859/memoria.pdf?sequence=1

> <a name="ref3">[3]</a> T. Gálvez Siuce, *Diseño de arquitectura de red neuronal convolucional para clasificación de imágenes*, Tesis de pregrado, Universidad Continental, 2023. [En línea]. Disponible en: https://repositorio.continental.edu.pe/bitstream/20.500.12394/14040/2/IV_FIN_113_TE_Galvez_Siuce_2023.pdf

> <a name="ref4">[4]</a> A. Comillas, *La capacidad predictiva de redes neuronales LSTM aplicadas a series de tiempo*, Tesis, Universidad Pontificia Comillas, 2023. [En línea]. Disponible en: https://repositorio.comillas.edu/rest/bitstreams/528011/retrieve

> <a name="ref5">[5]</a> Colaboración SciELO Chile, “Una comparación empírica de algoritmos de aprendizaje profundo para clasificación textual,” *Revista Chilena de Computación e Informática*, vol. 20, no. 2, 2022. [En línea]. Disponible en: https://www.scielo.cl/scielo.php?pid=S0718-33052022000200403&script=sci_arttext

> <a name="ref6">[6]</a> A. Richaud, *Algoritmos de optimización para redes neuronales*, Informe técnico, 2021. [En línea]. Disponible en: https://antonio-richaud.com/biblioteca/archivo/Algoritmos-de-optimizacion-para-RN/Algoritmos-de-optimizacion-para-RN.pdf
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
