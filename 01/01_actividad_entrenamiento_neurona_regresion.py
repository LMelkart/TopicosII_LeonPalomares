import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Actividad 1 · Entrenamiento paso a paso de una neurona artificial

    **Deep Learning para Computer Vision**

    **Nombre:** Leon Amilcar Palomares Magaña.

    ## Objetivo
    Implementar y explicar el entrenamiento de una neurona para regresión con
    NumPy, desde una predicción inicial hasta la actualización de sus parámetros.
    Al finalizar, deberás relacionar las ecuaciones, las dimensiones de los
    arreglos y el comportamiento de la pérdida durante el aprendizaje.

    Usaremos una activación identidad: este modelo corresponde a una regresión
    lineal. En clase lo abordamos como una primera aproximación al perceptrón.

    ## Instrucciones y entrega
    - Completa las celdas con `TODO` y las respuestas marcadas con **Tu respuesta**.
    - Implementa el modelo con NumPy. No utilices modelos de scikit-learn,
      TensorFlow, PyTorch ni herramientas de diferenciación automática.
    - En cada paso incluye explicación, ecuación, código e interpretación.
    - Conserva la convención de ejemplos en las **columnas** de `X`.
    - Entrega este archivo `.py` de marimo con tus respuestas, código y gráficas.
      Debe ejecutarse completo sin errores y permitir reproducir los resultados.
    - Los bloques incompletos devuelven `None` para que puedas abrir la plantilla.
      Sustitúyelos por tu implementación; no dejes resultados escritos a mano
      como reemplazo del código.

    Esta actividad estudia la mecánica del entrenamiento. Los datos de trabajo
    se usan para entrenar; no se evaluará generalización a datos nuevos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problema y datos

    Predeciremos un **puntaje sintético de rendimiento** a partir de dos
    características numéricas ya preparadas. Estos datos son didácticos y no
    representan mediciones reales. No requieren limpieza ni estandarización.

    1. Identifica las entradas y la salida del modelo.
    2. ¿Por qué este problema es de regresión?
    3. ¿Qué significa un ejemplo de entrenamiento en esta tabla?

    **Respuesta:**

    ***1*** Las variables de entrada son "Característica 1" y "Característica 2". Que son variables númericas. Mientras que la variable de salida sería "Puntaje"

    ***2*** Es una actividad de regresión debido a distintas razones. En primer lugar no hay ningún intento de crear un proceso de clasificación en 2 o más clases. Sino en la predicción de valores y el ajuste a la información ya disponible. En este caso, la predicción del puntaje, a partir de las características.

    ***3*** Sería los datos o características individuales de un registro, por ejemplo los primeros valores de *Característica 1* y *Característica 2*, que en conjunto dan como resultado el valor contenido en *Puntaje* del mismo registro. Es decir, serían los casos y la información que el perceptrón utilizará para aprender la relación entre ambos elementos, para producir el último.
    """)
    return


@app.cell
def _(np):
    X_datos = np.array([
        [-1.0, -0.5, 0.0, 0.5, 1.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, 0.5, -0.5, -1.0, 1.0, 1.0],
    ], dtype=float)
    y_datos = np.array([
        [-0.7, -0.9, 1.3, 1.0, 2.9, 0.6, -0.4, 1.2]
    ], dtype=float)
    return X_datos, y_datos


@app.cell
def _(X_datos, mo, y_datos):
    mo.ui.table([
        {"Ejemplo": i + 1, "Característica 1": float(X_datos[0, i]),
         "Característica 2": float(X_datos[1, i]), "Puntaje": float(y_datos[0, i])}
        for i in range(X_datos.shape[1])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dimensiones e inicialización

    Usaremos $m$ ejemplos, $n_x$ características y una salida:

    | Arreglo | Forma |
    |---|---|
    | $X$ | $(n_x,m)$ |
    | $y$ | $(1,m)$ |
    | $W$ | $(1,n_x)$ |
    | $b$ | $(1,1)$ |
    | $\hat y$ | $(1,m)$ |

    Completa `layer_sizes` e `initialize_parameters`. Inicializa pesos pequeños
    con `np.random.default_rng(seed)` y el sesgo en cero. Para esta neurona,
    inicializar en cero también sería posible; usaremos una semilla para
    comparar experimentos de forma reproducible.

    **Tu respuesta:** indica las dimensiones concretas y explica por qué
    el producto $WX$ y la suma del sesgo son válidos. ¿Qué controla la semilla?

    ***Dimensiones***

    | Arreglo | Forma |
    |---|---|
    | $X$ | $(2, 8)$ |
    | $y$ | $(1, 8)$ |
    | $W$ | $(1, 2)$ |
    | $b$ | $(1,1)$ |
    | $\hat y$ | $(1,8)$ |

    El producto $WX$es válido debido a las dimensiones de ambas matrices.

    $$A_(m*n) * B_(n*p) = C_(m*p)$$

    Las primeras tienen dimensiones $(1,2)$ y las segunda $(2,8)$. Lo cual da como resultado una matriz de dimension $(1,8)$.

    Esta matriz, al sumarla con el valor $b$ $(1,1)$, modifica cada una de los 8 registros sin tener errores de dimensionalidad.
    Por lo que el producto $WX$ + $b$ sigue manteniendo las dimensiones $(1,8)$


    La semilla seed() controla el punto de inicio al momento de crear un numero aleatorio. De esta forma, cada que se realiza el experimento puede ser reproducible. Esto porque los valores de inicio pueden cambiar mucho dependiendo del momento en el que es ejecutado el código.
    """)
    return


@app.cell
def _(np):
    def layer_sizes(X, y):
        n_x = X.shape[0]
        n_y = y.shape[0]
        return n_x, n_y

    def initialize_parameters(n_x, n_y, seed=42):
        W = np.random.randn(n_y, n_x)*0.1
        b = np.random.randn(n_y, 1)*0.01
        parameters = {"W": W, "b": b}
        return parameters

    return initialize_parameters, layer_sizes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Propagación hacia adelante

    $$Z=WX+b,\qquad \hat y=Z$$

    Implementa `forward_propagation`. Recibe los datos y un diccionario de
    parámetros; devuelve un arreglo con todas las predicciones.

    **Tu respuesta:** ¿qué aportan los pesos y el sesgo? ¿Por qué usamos
    la función identidad para la salida de este ejercicio?

    ***Los pesos $W$*** representan la importancia o el valor que tiene cada característica de entrada sobre el resultado final. Es decir si $w_1$ es grande el aumento o decremento de la *Característica 1* influirá considerablemente respecto al puntaje predicho. Mientras que si el valor es cercano a **$0$** tal característica casi no influirá en las decisiones de la neurona. Por tanto, ajustar los pesos es algo que nos permite indicar al perceptrón a qué debe prestar atención.

    En cambio ***los sesgos $b$*** funcionan como elementos o términos independientes de los datos de entrada. Pero que igual desviará o desplazara la función de activación hacia la izquierda o derecha. Esto permite que la neurona se active, aún si el valor de entrada es $0$

    La ***función de identidad*** permite que la salida del perceptrón entregue el resultado final de todas las ecuaciones hechas antes, sin modificarlo o transformarlo.
    """)
    return


@app.cell
def _(np):
    def forward_propagation(X, parameters):
        W = parameters["W"]
        b = parameters["b"]

        Z = np.matmul(W,X) + b
        y_hat = Z
        return y_hat

    return (forward_propagation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Función de pérdida

    $$J(W,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2$$

    Implementa `compute_cost`: debe devolver un escalar. Utiliza `np.sum`;
    si aparecen valores no finitos, investiga su causa.

    **Tu respuesta:** ¿por qué elevamos el error al cuadrado? ¿Qué efecto
    tiene el factor $1/2$ en la derivada? ¿Pérdida cero implica siempre
    que el modelo funcionará bien con datos nuevos?

    ***Elevación al cuadrado***
    Esto exita que el calculo de los errores se anulen entre sí, si estos tienen valores positivos y negativos distintos. De este modo, todos los errores se mantienen 'positivos'. Así mismo, permite detectar errores demasiado grandes. Por lo que nos ayuda a detectar predicciones que deben ser optimizadas.

    ***el factor $1/2$*** se utiliza debido a que al momento de derivar, en el paso siguiente, permite que el ajuste o resultado de la derivada del siguiente paso, sea valor 1, en lugar del 2 que resulta de la derivada de $()^2$. De esta forma el entrenamiento es directo y proporcional al error.

    ***Una pérdida 0*** no implica que funcione bien con datos nuevos. Esto podría ser resultado de que el perceptrón memorizo cada ejemplo que se le dió o un sobreajuste.
    """)
    return


@app.cell
def _(np):
    def compute_cost(y_hat, y):
        m = y_hat.shape[1]

        print(((y_hat - y)** 2.0))

        cost = np.sum((y_hat-y)**2.0) / (2*m)

        return cost

    return (compute_cost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Gradientes y actualización

    Con pesos organizados como **fila**, las ecuaciones son:

    \[
    dZ=\hat y-y,\qquad dW=\frac{1}{m}dZ X^T,
    \qquad db=\frac{1}{m}\sum_{i=1}^{m}dZ^{(i)}\]

    \[W_{\mathrm{nuevo}}=W-\alpha dW,\qquad
    b_{\mathrm{nuevo}}=b-\alpha db\]

    Implementa las dos funciones. Conserva las formas de `dW` y `db`;
    `np.sum(..., axis=1, keepdims=True)` puede ayudarte con el sesgo.
    La actualización debe devolver un diccionario nuevo sin modificar
    los arreglos originales en el lugar.

    **Tu respuesta:** muestra la derivación de $\partial J/\partial W_j$
    y $\partial J/\partial b$. Explica por qué restamos el gradiente y
    qué representa la tasa de aprendizaje $\alpha$.


    $$J(W,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2$$

    $$\hat{y} = XW + b$$

    $$\hat{y}^{(i)} = x_1^{(i)}W_1 + x_2^{(i)}W_2 + \dots + x_n^{(i)}W_n + b$$


    ***Derivación respecto al sesgo ($b$)***

    Aplicando la regla de la cadena para el sesgo:

    $$\frac{\partial J}{\partial b} = \frac{\partial}{\partial b} \left[ \frac{1}{2m}\sum_{i=1}^{m}(\hat y^{(i)}-y^{(i)})^2 \right]$$

    $$\frac{\partial J}{\partial b} = \frac{1}{2m}\sum_{i=1}^{m} 2(\hat y^{(i)}-y^{(i)}) \cdot \frac{\partial \hat y^{(i)}}{\partial b}$$

    Como la derivada de $\hat{y}^{(i)}$ respecto al sesgo $b$ es $1$ para cada ejemplo:

    $$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot 1$$

    $$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} dZ^{(i)}$$

    ***Derivacion respecto a los pesos ($W$)***

    $$\hat{y}^{(i)} = x_1^{(i)}W_1 + x_2^{(i)}W_2 + \dots + x_n^{(i)}W_n + b$$

    $$\frac{\partial \hat{y}^{(i)}}{\partial W_j} = x_j^{(i)}$$

    Como $X$ y $W$ son matrices, multiplicandose, al momento de derivar W, tenemos la matriz X, pero con la estructura $(8, 2)$
    $$\frac{\partial \hat{y}^{(i)}}{\partial W} = x^{(i)}$$

    $$\frac{\partial J}{\partial W} = \frac{1}{2m} \sum_{i=1}^{m} 2(\hat{y}^{(i)} - y^{(i)}) \cdot \frac{\partial \hat{y}^{(i)}}{\partial W}$$

    $$\frac{\partial J}{\partial W} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$

    $$(m \times n) \times (m \times 1)$$

    $$(n \times m) \times (m \times 1) = (n \times 1)$$

    $$\frac{\partial J}{\partial W} = \frac{1}{m} X^T (\hat{y} - y)$$

    $$\frac{\partial J}{\partial W} = \frac{1}{m}\sum_{i=1}^{m} (\hat y^{(i)}-y^{(i)}) X^{(i)T}$$

    Dado que la operación hacia adelante es $\hat{y} = WX + b$, al expresar los errores colectivamente como $dZ = \frac{1}{m}(\hat{y} - y)$ (de dimensión $1 \times m$), y para que el gradiente $dW$ tenga la misma forma que $W$ ($1 \times n_x$), transponemos $X$ ($X^T$ de $m \times n_x$):

    $$\frac{\partial J}{\partial W} = \frac{1}{m} dZ X^T$$

    Restamos el gradiente porque este indica la dirección optima donde la pérdida sería mayor. Y debido a que lo que queremos es lo opuesto, o encontrar el punto más bajo de error y minimizar la perdida, se resta en lugar de sumar.

    Así mismo, la tasa de aprendizaje $$\alpha$$ es un parámetro que indica el tamaño que se estarán dando los *pasos* cada iteración. Si este valor es muy grande podemos errar respecto al error mínimo, o desestabilizar el proceso de aprendizaje. Pero, si es muy pequeña, el modelo podría tardar más de lo necesario en aprender.
    """)
    return


@app.cell
def _(np):
    def back_propagation(y_hat, y, X):
        m = X.shape[1]
        dZ = y_hat - y
        dW = (1/m) * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        grads = {"dW":dW, "db":db}
        return grads

    def update_parameters(parameters, grads, learning_rate):
        W = parameters["W"]
        b = parameters["b"]

        dW = grads["dW"]
        db = grads["db"]

        W = W - learning_rate * dW
        b = b - learning_rate * db

        parameters = {'W':W, 'b':b }
        return parameters

    return back_propagation, update_parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Una iteración a mano y con NumPy

    Utiliza exclusivamente los siguientes dos ejemplos y parámetros:

    $$X_{\mathrm{manual}}=\begin{bmatrix}1&2\\0&1\end{bmatrix},
    \quad y_{\mathrm{manual}}=\begin{bmatrix}2&3\end{bmatrix}$$
    $$W_0=\begin{bmatrix}0&0\end{bmatrix},\quad b_0=0,\quad\alpha=0.1$$

    **Ejercicio:**, desarrolla a mano las predicciones iniciales,
    la pérdida, `dZ`, `dW`, `db`, los parámetros actualizados y la pérdida
    después de la actualización. Escribe las operaciones, no solo el resultado.

    **Tu desarrollo:**

    ***Predicciones iniciales***

    $$\hat y = WX + b$$
    $$\hat y = \begin{bmatrix}0&0\end{bmatrix} \begin{bmatrix}1&2\\0&1\end{bmatrix} + 0 = \begin{bmatrix} (1)(0) + (0)(0) & (2)(0) + (1)(0) \end{bmatrix} = \begin{bmatrix}0&0\end{bmatrix}$$

    ***Pérdida inicial ($J$)***

    Utilizando la función de pérdida cuadrática:
    $$J = \frac{1}{2(2)} \sum_{i=1}^{2} (\hat y_i - y_i)^2$$

    Calculamos la diferencia $\hat{y}_0 - y$:
    $$\hat{y}_0 - y = \begin{bmatrix}0&0\end{bmatrix} - \begin{bmatrix}2&3\end{bmatrix} = \begin{bmatrix}-2&-3\end{bmatrix}$$

    Sustituyendo en la pérdida:
    $$J_0 = \frac{1}{4} \left( (-2)^2 + (-3)^2 \right) = \frac{1}{4} (4 + 9) = \frac{13}{4} = 3.25$$

    ***Cálculo de gradientes con división por $m$***

    Al derivar la función de costo con el factor $\frac{1}{m}$, los gradientes se dividen entre $m = 2$:

    * **`dZ`** (Derivada respecto a la salida, escalada por $1/m$):
      $$\mathrm{dZ} = \frac{1}{m} (\hat{y}_0 - y) = \frac{1}{2} \begin{bmatrix}-2&-3\end{bmatrix} = \begin{bmatrix}-1&-1.5\end{bmatrix}$$

    $dW$ *(Derivada respecto a los pesos):*

    $$\mathrm{dW} = \mathrm{dZ} X^T$$
    $$\mathrm{dW} = \begin{bmatrix} -1 & -1.5 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$$
    $$\mathrm{dW} = \begin{bmatrix} (-1)(1) + (-1.5)(2) & (-1)(0) + (-1.5)(1) \end{bmatrix} = \begin{bmatrix} -4 & -1.5 \end{bmatrix}$$

    * **`db`** (Derivada respecto al sesgo):
      $$\mathrm{db} = \sum_{i=1}^{2} \mathrm{dZ}_i = -1 + (-1.5) = -2.5$$

    ***Actualización de parámetros ($W_1$, $b_1$)***

    Usando la tasa de aprendizaje $\alpha = 0.1$:

    * **Nuevos pesos ($W_1$):**
      $$W_1 = W_0 - \alpha \mathrm{dW}$$
      $$W_1 = \begin{bmatrix} 0 & 0 \end{bmatrix} - 0.1 \begin{bmatrix} -4 & -1.5 \end{bmatrix} = \begin{bmatrix} 0.4 & 0.15 \end{bmatrix}$$

    * **Nuevo sesgo ($b_1$):**
      $$b_1 = b_0 - \alpha db = 0 - 0.1(-2.5) = 0.25$$

    ***5. Predicciones y pérdida después de la actualización***

    * **Nuevas predicciones ($\hat{y}_1$):**
    $$W_1 = \begin{bmatrix} 0.4 & 0.15 \end{bmatrix}$$
    $$\hat{y}_1 = W_1 X + b_1 = \begin{bmatrix} 0.4 & 0.15 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}  + 0.25$$
    $$\hat{y}_1 = \begin{bmatrix} (0.4)(1) + (0.15)(0) & (0.4)(2) + (0.15)(1) \end{bmatrix} + 0.25$$
    $$\hat{y}_1 = \begin{bmatrix} 0.4 & 0.95 \end{bmatrix} + 0.25 = \begin{bmatrix} 0.65 & 1.20 \end{bmatrix}$$

    * **Nueva pérdida ($J_1$):**
    $$\hat{y}_1 - y = \begin{bmatrix} 0.65 & 1.20 \end{bmatrix} - \begin{bmatrix} 2 & 3 \end{bmatrix} = \begin{bmatrix} -1.35 & -1.80 \end{bmatrix}$$
    $$J_1 = \frac{1}{4} \left( (-1.35)^2 + (-1.80)^2 \right)$$

    $$J_1 = \frac{1}{4} (1.8225 + 3.24) = \frac{1}{4} (5.0625) = 1.265625$$

    Después, completa la celda para repetir el cálculo usando tus funciones.
    Muestra los valores y comprueba que coinciden con tu
    desarrollo manual. Explica cualquier diferencia por redondeo.

    **Tu interpretación:** ¿disminuyó la pérdida? ¿Cómo cambiaron las predicciones?

    Sí, la perdida disminuyó considerablemente. Lo que nos indica que si bien las predicciones son más cercanas a los valores mínimos, no necesariamente es el resultado es el más optimo, pero si se encuentra más cercano que con los valores iniciales.
    """)
    return


@app.cell
def _(
    back_propagation,
    compute_cost,
    forward_propagation,
    np,
    update_parameters,
):
    X_manual = np.array([[1.0, 2.0], [0.0, 1.0]])
    y_manual = np.array([[2.0, 3.0]])
    parametros_manual = {"W": np.zeros((1, 2)), "b": np.zeros((1, 1))}
    alpha_manual = 0.1

    print(f"W inicial: {parametros_manual['W']}")
    print(f"b inicial: {parametros_manual['b']}")

    y_hat_manual = forward_propagation(X_manual, parametros_manual)
    costo_inicial = compute_cost(y_hat_manual, y_manual)
    print(f"Predicción inicial (y_hat): {y_hat_manual}")
    print(f"Costo inicial (J): {costo_inicial:.6f}")

    grads_manual = back_propagation(y_hat_manual, y_manual, X_manual)
    print(f"Gradiente W (dW): {grads_manual['dW']}")
    print(f"Gradiente b (db): {grads_manual['db']}")


    parametros_nuevos = update_parameters(parametros_manual, grads_manual, learning_rate=alpha_manual)
    print(f"W nuevo: {parametros_nuevos['W']}")
    print(f"b nuevo: {parametros_nuevos['b']}")


    y_hat_nuevo = forward_propagation(X_manual, parametros_nuevos)
    costo_nuevo = compute_cost(y_hat_nuevo, y_manual)
    print(f"Nueva predicción (y_hat): {y_hat_nuevo}")
    print(f"Nuevo costo (J): {costo_nuevo:.6f}")
    print(f"¿El costo disminuyó?: {costo_nuevo < costo_inicial}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Ciclo de entrenamiento

    Integra tus funciones en `train_model`:

    1. Obtén las dimensiones usando los argumentos `X` e `y`.
    2. Inicializa los parámetros una sola vez.
    3. Guarda la pérdida inicial.
    4. En cada iteración: predice, calcula gradientes, actualiza parámetros
       y guarda la pérdida **después** de actualizar.
    5. Devuelve los parámetros finales y el historial de pérdidas.

    El historial debe tener `iterations + 1` valores: el índice cero es
    la pérdida antes de entrenar. Usa únicamente los datos recibidos por
    la función, para que también funcione con el conjunto del cálculo manual.

    Entrena con 200 iteraciones, tasa 0.1 y semilla 42. Muestra los parámetros,
    las predicciones finales, las pérdidas inicial y final y una gráfica
    de pérdida contra número de actualizaciones.

    **Tu respuesta:** describe qué se repite y qué se conserva en cada iteración.
    ¿Una iteración de este entrenamiento utiliza todos los ejemplos?

    ***Se repite*** el flujo completo de propagación hacia adelante ('forward_propagation'), el calculo de costo, la retropropagación para obtener los gradientes ('back_propagation') y la aplicación del descenso de gradiente para actualizar los pesos y el sesgo ('update_parameters').
    ***Se conserva*** los datos de entrada y de salida ($X$, $y$), la tasa de aprendizaje $\alpha$, y los valores actualizados de los parametros $W$ y $b$ que se heredan y recalculan en la siguiente iteración o época.

    Una iteración de entrenamiento si implementa todos los ejemplos **$m$**, contenidos en $X$ y $y$ para calcular el gradiente global, y así actualizar parámetros, para su posterior reproducción de nueva cuenta.
    """)
    return


@app.cell
def _(
    X_datos,
    back_propagation,
    compute_cost,
    forward_propagation,
    initialize_parameters,
    layer_sizes,
    update_parameters,
    y_datos,
):
    def train_model(X, y, iterations=200, learning_rate=0.1, seed=42):
        n_x, n_y = layer_sizes(X=X_datos, y=y_datos)

        parameters = initialize_parameters(n_x=n_x, n_y=n_y, seed=seed)
        cost_history = []
        iter_list = []

        y_hat = forward_propagation(X=X_datos, parameters=parameters)
        costo_inicial = compute_cost(y_hat=y_hat, y=y_datos)
        cost_history.append(costo_inicial)
        iter_list.append(0)

        for i in range(iterations):
            y_hat = forward_propagation(X=X_datos, parameters=parameters)
            grads = back_propagation(y_hat=y_hat, y=y_datos, X=X_datos)
            parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)
            y_hat = forward_propagation(X=X_datos, parameters=parameters)
            costo_inicial = compute_cost(y_hat=y_hat, y=y_datos)
    
            cost_history.append(costo_inicial)
            iter_list.append(i+1)

        return parameters, cost_history, iter_list
        return (train_model,)

    return (train_model,)


@app.cell
def _(X_datos, plt, train_model, y_datos):
    parameters, cost_history, iter_list = train_model(
            X=X_datos, 
            y=y_datos, 
            iterations=200, 
            learning_rate=0.1,
            seed=42
        )

    print(f"Parámetros finales W: {parameters['W']}")
    print(f"Parámetro final b: {parameters['b']}")
    print(f"Pérdida inicial: {cost_history[0]:.6f}")
    print(f"Pérdida final: {cost_history[-1]:.6f}")

        # Graficar el historial de pérdidas
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iter_list, cost_history, color='royalblue', linewidth=2)
    ax.set_title("Evolución de la Función de Costo J(W, b)")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Costo (Pérdida)")
    ax.grid(True, linestyle='--', alpha=0.6)

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Experimento con la tasa de aprendizaje

    Repite el entrenamiento con $\alpha\in\{0.01,0.1,3.0\}$.
    Usa los mismos datos, **200 iteraciones y la misma semilla** en todos
    los casos. Reinicia el entrenamiento del modelo para cada experimento.

    - Grafica las tres curvas, con leyenda y ejes etiquetados. Puedes usar
      escala logarítmica o paneles separados si las magnitudes lo requieren.
    - Construye una tabla con tasa, pérdida inicial, pérdida final y
      si observaste convergencia, oscilaciones o divergencia.
    - Si aparecen valores no finitos, indícalos y explica lo ocurrido.

    **Tu análisis:** ¿qué tasa elegirías entre estas opciones y por qué?
    ¿Por qué una tasa mayor no garantiza aprender mejor? ¿Qué puedes
    concluir con solo 200 iteraciones y qué no puedes afirmar todavía?

    Elegiría $\alpha = 0.1$.
    Esto porque ofrece mejor equilibrio entre la velocidad y estabilidad del proceso. Debido a que cuando $\alpha=0.01$ avanza de forma muy lenta, y apun con 200 iteraciones no logra llegar a un valor optimo, por lo que el calculo seria mucho mas largo.

    Asi mismo $\alpha = 0.3$  es una tasa que como mencione antes incrementa el paso con el que se desplaza el modelo, pero, debido a su escala sobrepasa el punto minimo de la funcion de costo. Esto puede ocasionar un sobreajuste, o un desbordamiento de los datos.

    Respecto a las 200 iteraciones, si la tasa de aprendizaje es adecuada, como la de $0.1$, el perceptron es capaz de minimizar el error de entrenamiento con pocas iteraciones, actualizando correctamente sus pesos y sesgo.

    Sin embargo, no se puede asegurar que el perceptron actue correctamente ante datos nuevos. Por lo que no se puede afirmar que 200 iteraciones sean suficientes en conjuntos mas grandes, donde se puedan requerir mas epocas o un ajuste distinto a la tasa de aprendizaje.
    """)
    return


@app.cell
def _(X_datos, plt, train_model, y_datos):
    def _():
        parameters, cost_history, iter_list = train_model(
                X=X_datos, 
                y=y_datos, 
                iterations=200, 
                learning_rate=0.01,
                seed=42
            )

        print(f"Parámetros finales W: {parameters['W']}")
        print(f"Parámetro final b: {parameters['b']}")
        print(f"Pérdida inicial: {cost_history[0]:.6f}")
        print(f"Pérdida final: {cost_history[-1]:.6f}")

    
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iter_list, cost_history, color='royalblue', linewidth=2)
        ax.set_title("Evolución de la Función de Costo J(W, b)")
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Costo (Pérdida)")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig


    _()
    return


@app.cell
def _(X_datos, plt, train_model, y_datos):
    def _():
        parameters, cost_history, iter_list = train_model(
                X=X_datos, 
                y=y_datos, 
                iterations=200, 
                learning_rate=0.1,
                seed=42
            )

        print(f"Parámetros finales W: {parameters['W']}")
        print(f"Parámetro final b: {parameters['b']}")
        print(f"Pérdida inicial: {cost_history[0]:.6f}")
        print(f"Pérdida final: {cost_history[-1]:.6f}")

    
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iter_list, cost_history, color='royalblue', linewidth=2)
        ax.set_title("Evolución de la Función de Costo J(W, b)")
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Costo (Pérdida)")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig


    _()
    return


@app.cell
def _(X_datos, plt, train_model, y_datos):
    def _():
        parameters, cost_history, iter_list = train_model(
                X=X_datos, 
                y=y_datos, 
                iterations=200, 
                learning_rate=0.3,
                seed=42
            )

        print(f"Parámetros finales W: {parameters['W']}")
        print(f"Parámetro final b: {parameters['b']}")
        print(f"Pérdida inicial: {cost_history[0]:.6f}")
        print(f"Pérdida final: {cost_history[-1]:.6f}")

    
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iter_list, cost_history, color='royalblue', linewidth=2)
        ax.set_title("Evolución de la Función de Costo J(W, b)")
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Costo (Pérdida)")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Conclusiones y conexión con Deep Learning

    Responde con referencia a tus resultados:

    1. ¿Qué significa que la neurona haya aprendido? Relaciona parámetros,
       predicciones y pérdida.
    2. ¿Qué diferencia existe entre calcular gradientes y actualizar parámetros?
    3. ¿Cómo comprobarías si una implementación entrega resultados correctos, además de observar que la pérdida disminuye?
    4. Si usamos sigmoide y entropía cruzada para clasificación binaria,
       ¿qué partes del proceso cambian y cuáles conservan su estructura?

    **Tus conclusiones:**

    ***1.*** Se dice que la neurona aprendio, cuando el perceptron logra encontrar una combinacion optima o se minimiza el error con la actualizacion de los parametros $W$ y $b$. Con esto, las predicciones se alinean con valores reales objetivo. Y como conse cuencia la perdida $J$ disminuye hasta acercarce a cero.

    ***2.*** los gradientes, nos indican hacia donde y con que intensidad el error es sensible a los parametros $W$ y $b$. Mientras que la actualizacion de los parametros, consiste en acercar los valores numericos reales, con base de la tasa de aprendizaje y el gradiente, en direccion del valor minimo que se esta buscando, por lo que se actualizan con frecuencia.

    ***3.*** Para comprobar esto, en primer lugar podemos evaluar como avanzan y se transforman los datos con las iteraciones, y como va modificando los paraametros de perdida. No obstante, en otro caso, al tener conjuntos de datos de entrenamiento y de prueba, los de prueba nos servirian para  detectar que tan preciso es el modelo que entrenamos a partir de valores como accuracy.

    ***4.*** Los valores que cambian, son el valor de identidad, que cambia por la funcion sigmoide, de forma que los datos de salida se acoten al rango $(0, 1)$ para que se transforme en una probabilidad de clase. Ademas de que el  uso de Entropia Cruzada Binaria, esta disenada para penalizar errores en decisiones de clasificacion.

    Mientras que conserva la estructura $Z = WX + b$, al igual que las formulas de gradientes. Y las reglas de actualizacion de pesos con $\alpha$.

    ## Rúbrica

    | Criterio | Valor | Evidencia esperada |
    |---|---:|---|
    | Implementación correcta | 40 % | Dimensiones coherentes, funciones propias, actualización y ciclo completos; coincidencia con la iteración manual. |
    | Explicación matemática | 30 % | Ecuaciones, derivación de gradientes y explicación del papel de cada paso. |
    | Experimentación e interpretación | 20 % | Comparación controlada de tasas, gráficas, tabla y conclusiones sustentadas. |
    | Claridad y reproducibilidad | 10 % | Notebook ordenado, semilla fija, respuestas completas y ejecución sin errores. |
    """)
    return


if __name__ == "__main__":
    app.run()
