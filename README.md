# Gestión de camas hospitalarias con ML — Predicción de LOS y Long-Stay

Proyecto de ciencia de datos para anticipar la **longitud de estancia (LOS)** y la **probabilidad de estancia prolongada (long-stay)** a fin de mejorar la planificación de camas, dotaciones y coordinación de altas/interconsultas. Enfoque operativo: maximizar **Recall@Top-N** (alertas útiles) y minimizar **MAE** (planificación de LOS).

---

## Cómo ejecutar

### Opción A — Google Colab (recomendada)
1) Abrir el **Colab** desde el enlace de arriba.  
2) Menú **Entorno > Reiniciar y ejecutar todo**.  
3) Si se prefiere usar el `.ipynb` local o alojado en otro Drive, en Colab usar:
   - **Archivo > Subir notebook** (para un `.ipynb` local), o
   - **Archivo > Abrir cuaderno > Google Drive** (para usar en Drive propio).

### Opción B — Local (Python 3.10+)
```bash
# Clonar
git clone https://github.com/diplocdaagrupon/TIF_GrupoN.git
cd TIF_GrupoN

# Crear y activar entorno
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -U pip
pip install -r requirements.txt
