# Resumen del Curso: Engineers Guide to Soar (Proyectos 0-4)

## Proyecto 0: Iniciando Soar (Starting Soar)
**Concepto:** Introducción al entorno.
Este capítulo no implica escribir lógica en Soar, sino configurar el entorno de desarrollo. El objetivo es ejecutar el script de Python (`run_project.py`) para lanzar el depurador de Soar (SoarJavaDebugger) y asegurarse de que el agente se inicia correctamente.

## Proyecto 1: Hola Mundo (Hello World)
**Concepto:** Estructura básica de una regla (Production).
Se introduce la sintaxis fundamental de Soar:
* **Reglas (`sp`):** La unidad básica de lógica.
* **LHS (Left Hand Side):** Condiciones (SI...). Aquí verificamos `(state <s> ^type state)`.
* **RHS (Right Hand Side):** Acciones (...ENTONCES).
* **Funciones:** Uso de `(write)` para imprimir en consola y `(crlf)` para salto de línea.

## Proyecto 2: Enlace de Entrada (Input Link)
**Concepto:** Percepción del entorno.
El agente deja de ser pasivo y comienza a leer datos externos a través de la estructura `io.input-link`.
* **Notación de puntos:** Acceso a estructuras anidadas (ej. `^io.input-link.candidate-supplier`).
* **Variables:** Uso de `<variable>` para hacer coincidir cualquier valor o atributo dinámico (ej. leer múltiples proveedores con una sola regla).
* **Carga de archivos:** Uso de comandos como `load file` y `cd`.

## Proyecto 3: Conceptos Básicos de Operadores (Operator Basics)
**Concepto:** Toma de decisiones y actuación interna.
El núcleo de la inteligencia de Soar. Se pasa de "reaccionar" (reglas de elaboración) a "decidir" usando el Ciclo de Decisión:
1.  **Proponer (Propose):** Sugerir acciones posibles (ej. "podría seleccionar al proveedor X").
2.  **Preferir (Prefer):** Ordenar las propuestas (ej. "el proveedor con mayor puntaje es mejor").
3.  **Aplicar (Apply):** Ejecutar la acción elegida y modificar el estado (memoria) para avanzar.

## Proyecto 4: Salida (Output)
**Concepto:** Actuación sobre el entorno.
Comunicar las decisiones del agente al mundo exterior a través de `io.output-link`.
* **Preferencias Unarias:** Uso de preferencias como "Worst" (`<`) para diferir acciones (ej. "solo genera la salida cuando hayas terminado todo lo demás").
* **Interrupción:** Uso de `(interrupt)` para detener al agente una vez finalizada la tarea.

## Proyecto 5: Lógica de Aplicación Múltiple (Multi-Apply)
**Concepto:** Descomposición de acciones y preservación del orden.
Se enseña cómo un solo operador puede tener múltiples reglas de aplicación (`apply rules`) que trabajan en conjunto para realizar acciones complejas de manera modular.
* **Problema del orden:** La salida en lote (batch) del capítulo anterior no garantizaba el orden de los proveedores.
* **Solución:** Mover la lógica de salida dentro del operador de selección (`select-supplier`). Al tener reglas de aplicación dedicadas que se disparan *durante* la selección de cada proveedor, el agente comunica su decisión al entorno paso a paso, garantizando que la lista de salida respete estrictamente el orden de preferencia calculado.

## Proyecto 6: Subestados (Substates)
**Concepto:** Resolución jerárquica de problemas.
Se introduce el manejo de **Impasses** (bloqueos en la toma de decisiones). Cuando el agente no puede decidir entre varios operadores (Operator Tie), Soar genera un subestado.
* **Estructura:** Uso de `^superstate` para acceder a la información del nivel superior desde el subestado.
* **Lógica:**
    1. El agente encuentra un empate.
    2. Entra en un subestado.
    3. Ejecuta lógica para comparar los ítems (`^item`) que causaron el empate.
    4. Genera una preferencia (`>`) en el superestado para romper el empate.
* **Beneficio:** Permite realizar razonamientos complejos o pasos intermedios sin "ensuciar" el flujo principal de decisión.

**Concepto:** Resolución jerárquica de bloqueos (Impasses).
Se utiliza cuando el agente tiene múltiples opciones válidas pero ninguna regla de preferencia directa para elegir entre ellas.
* **Proceso:** Soar genera un subestado de tipo `tie`. El agente entra en este subestado, evalúa los ítems en conflicto basándose en criterios específicos (como `total-score`) y devuelve una preferencia de orden al superestado.
* **Resultado:** El impasse se resuelve y el flujo principal continúa de manera ordenada.

---

## Proyecto 7: Depuración y Clasificación Personalizada
**Concepto:** Iteración sobre atributos dinámicos.
El agente utiliza lógica de subestados para iterar sobre los pesos de prioridad definidos en el `input-link`.
* **Lógica:** Selecciona el peso más alto no evaluado -> Ordena proveedores por el atributo asociado a ese peso -> Marca el peso como evaluado -> Repite.
* **Depuración:** Uso de `matches -w` para entender por qué una regla no se dispara.

## Proyecto 8: Organización y Subtotales
**Concepto:** Manejo de empates complejos y estructura de código.
Se aborda el problema de cuando múltiples atributos tienen la misma prioridad.
* **Solución:** Un subestado anidado (`suppliersort-find-subtotal`) suma los puntajes de los atributos empatados y devuelve un valor combinado para permitir la clasificación.

## Proyecto 9: Combinación de Preferencias (Reject)
**Concepto:** Filtrado y restricciones.
Uso de la preferencia de "Rechazo" (`-`) para descartar candidatos que no cumplen criterios mínimos (ej. `total-sats` es 0) o para limitar la cantidad de resultados (ej. "Top 3"). También implementa la limpieza del agente (`init`) tras generar la salida.

---

# **SUPEREXTRA**

## Proyecto 10: Conceptos Básicos de SML
**Concepto:** Interfaz Python-Soar.
Uso de la librería `sml` para crear un Kernel, cargar agentes (`LoadProductions`), crear estructuras de entrada (`CreateIntWME`, etc.) y leer la salida recorriendo el grafo de WMEs.

## Proyecto 11: Gestión de WMEs
**Concepto:** Ciclo de vida de la entrada.
Métodos para limpiar la memoria de trabajo (`DestroyWME`) y relanzar el debugger desde código (`SpawnDebugger`), permitiendo múltiples ejecuciones secuenciales.

## Proyecto 12: Gestión de I/O Orientada a Objetos
**Concepto:** Abstracción.
Creación de clases Python que envuelven la lógica de SML para añadir/borrar proveedores, haciendo el código principal mucho más limpio y escalable.

## Proyecto 13: Eventos (Event Handlers)
**Concepto:** Programación reactiva.
Uso de callbacks para reaccionar a sucesos del agente:
* `RegisterForPrintEvent`: Capturar `(write)`.
* `AddOutputHandler`: Detectar cambios específicos en el `output-link`.
* `RegisterForRunEvent`: Ejecutar lógica de Python (ej. inyectar input) justo antes de que Soar empiece a pensar.