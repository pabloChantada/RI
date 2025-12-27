import sys
import os
'''
ESTO DE AQUI DE MOMENTO NO FUNCIONA, SI NOS DICE QUE NO HACE FALTA HACERLO NOS LO CARGAMOS
'''
# --- CONFIGURACIÓN DE RUTAS ---
# Ajusta esta ruta para que apunte a donde tienes el binario de Soar si no se encuentra automáticamente
# sys.path.append("/path/to/Soar/bin") 

try:
    import Python_sml_ClientInterface as sml
except ImportError:
    print("Error: No se encuentra la librería 'Python_sml_ClientInterface'.")
    print("Asegúrate de que el 'soar/bin' esté en tu PYTHONPATH.")
    sys.exit(1)

# Ruta a tu archivo principal del agente (el que carga todo)
# Asumimos que está en la misma carpeta que este script o ajusta la ruta.
AGENT_PATH = "suppliersort_main.soar" 

# --- CLASES AUXILIARES (Capítulo 12: IO Objects) ---

class InputSupplier:
    """Clase para facilitar la creación de proveedores en el Input-Link"""
    def __init__(self, name, cost, sats, scores):
        self.name = name
        self.cost = cost
        self.sats = sats
        self.scores = scores # Diccionario con puntajes: {'quality': 10, ...}
        self.wme_id = None # Identificador en Soar

    def add_to_soar(self, input_link):
        # Crear estructura ^candidate-supplier
        self.wme_id = input_link.CreateIdWME("candidate-supplier")
        self.wme_id.CreateStringWME("name", self.name)
        self.wme_id.CreateFloatWME("total-cost", self.cost)
        self.wme_id.CreateIntWME("total-sats", self.sats)
        
        # Añadir puntajes individuales (para los capítulos de pesos)
        # Nota: Ajusta estos campos según lo que tu agente espere leer
        self.wme_id.CreateIntWME("total-score", sum(self.scores.values()))
        for attr, val in self.scores.items():
            self.wme_id.CreateIntWME(attr, val)

    def remove_from_soar(self):
        if self.wme_id:
            self.wme_id.DestroyWME()
            self.wme_id = None

# --- CALLBACKS (Capítulo 13: Eventos) ---

def on_print_event(id, user_data, agent, message):
    """Captura lo que el agente escribe con (write)"""
    print(f"[SOAR]: {message.strip()}")

def on_output_event(id, user_data, agent, attribute_name, wme):
    """Detecta cuando el agente genera output"""
    if attribute_name == "supplier-list":
        print("\n>>> ¡OUTPUT DETECTADO! El agente ha enviado una lista. <<<")
        # Aquí podríamos leer la lista recorriendo el WME...

# --- MAIN ---

def main():
    # 1. CREAR KERNEL Y AGENTE (Cap 10)
    kernel = sml.Kernel.CreateKernelInNewThread()
    if not kernel or kernel.HadError():
        print("Error creando Kernel:", kernel.GetLastErrorDescription())
        return

    agent = kernel.CreateAgent("SoarAgent")
    if not agent:
        print("Error creando Agente:", kernel.GetLastErrorDescription())
        return

    # 2. CARGAR REGLAS
    if not os.path.exists(AGENT_PATH):
        print(f"Error: No encuentro el archivo {AGENT_PATH}")
        return
    
    print(f"Cargando agente desde: {AGENT_PATH}")
    agent.LoadProductions(AGENT_PATH)

    # 3. REGISTRAR EVENTOS (Cap 13)
    agent.RegisterForPrintEvent(sml.smlEVENT_PRINT, on_print_event, None)
    # Escuchar cambios en el output-link
    agent.AddOutputHandler("supplier-list", on_output_event, None)

    # 4. PREPARAR INPUT (Cap 12)
    input_link = agent.GetInputLink()
    
    # Definir prioridades (Input Link global)
    priorities = input_link.CreateIdWME("priorities")
    priorities.CreateFloatWME("total-cost", 11.01)
    priorities.CreateIntWME("quality", 11)
    
    # Configuración
    settings = input_link.CreateIdWME("settings")
    settings.CreateIntWME("max-output-suppliers", 3)

    # Crear proveedores usando nuestra clase Python
    suppliers = [
        InputSupplier("supplier01", 50.0, 2, {'quality': 10}),
        InputSupplier("supplier02", 20.0, 2, {'quality': 8}), # Barato
        InputSupplier("supplier03", 100.0, 0, {'quality': 5}) # Sats 0 -> Debería ser rechazado
    ]

    print("\n--- Inyectando Input ---")
    for s in suppliers:
        s.add_to_soar(input_link)

    # 5. EJECUTAR AGENTE
    print("\n--- Ejecutando Soar ---")
    agent.RunSelfForever() # Correrá hasta que el agente haga (halt) o (interrupt)

    # 6. LIMPIEZA
    kernel.Shutdown()
    print("\nSoar finalizado.")

if __name__ == "__main__":
    main()