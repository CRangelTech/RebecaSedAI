
# 🌙✨ Rebeca SedAI — Plugin IA generador de Wireframes para Figma

---

## 📌 Descripción

**Rebeca Sedai** es un plugin experimental para **Figma**, diseñado para transformar **descripciones textuales (prompts)** en **wireframes de complejidad media**.  
Su objetivo es democratizar la creación de interfaces de alta fidelidad para **usuarios sin conocimientos técnicos**, usando una red neuronal **entrenada desde cero**, sin APIs ni modelos preentrenados de terceros.

Este proyecto forma parte de un **trabajo de investigación académica**, explorando:
- **Procesamiento de lenguaje natural aplicado a UI.**
- **Conversión texto → estructura de nodos Figma.**
- **Ejemplos didácticos para futuros estudiantes.**

---

## 🎯 Objetivos

- Convertir **prompts en lenguaje natural**, separados por comas, en **layouts de UI organizados automáticamente**.
- Entrenar una **red neuronal propia** con un dataset derivado de wireframes reales exportados desde Figma.
- Ejecutar la inferencia **offline**, dentro del plugin, usando `TensorFlow.js`.
- Evitar que el usuario final tenga que configurar servidores o correr scripts externos.
- Servir como ejemplo de **arquitectura pedagógica** para IA aplicada a herramientas de diseño.

---

## 🗂️ Estructura del proyecto

```
📦 Rebeca Sedai [PROYECTO PRINCIPAL]
├── 📄 .gitignore [archivos que ignorará Git]
├── 📄 README.md [documentación principal]
├── 📂 TRAINING [entrenamiento de modelos]
│   ├── 📄 train.py [script de entrenamiento]
│   ├── 📄 export_onnx.py [exportación a ONNX]
│   ├── 📄 convert_tfjs.py [conversión a TensorFlow.js]
│   ├── 📄 requirements.txt [dependencias]
│   ├── 📂 experiments [experimentos]
│   └── 📂 data [datos]
│       ├── 📂 raw [datos crudos]
│       └── 📂 processed [datos procesados]
├── 📂 PLUGIN [extensión principal]
│   ├── 📄 manifest.json [configuración]
│   ├── 📄 package.json [dependencias NPM]
│   ├── 📄 tsconfig.json [config TypeScript]
│   ├── 📄 code.ts [lógica principal]
│   ├── 📄 controller.ts [controlador]
│   ├── 📂 ui [interfaz]
│   │   ├── 📄 index.html [entrada HTML]
│   │   ├── 📄 App.tsx [componente principal]
│   │   └── 📂 components [componentes React]
│   ├── 📂 model [modelo AI]
│   │   ├── 📄 model.json [arquitectura]
│   │   └── 📄 weights.bin [pesos entrenados]
│   └── 📂 utils [utilidades]
│       ├── 📄 tokenizer.ts [procesamiento texto]
│       └── 📄 mapper.ts [mapeo datos]
└──  📂 docs [documentación]
    ├── 🖼️ diagrama_arquitectura.png [diagrama sistema]
    └── 📝 flujo_inferencia.md [explicación inferencia]
```

---

## ⚙️ Tecnologías clave

- **Figma Plugin API** — Manipulación de nodos y canvas.
- **React + TypeScript** — Interfaz del plugin.
- **TensorFlow.js** — Inferencia del modelo dentro del navegador.
- **PyTorch** — Entrenamiento local del modelo.
- **ONNX** — Puente para exportar modelo PyTorch a JS.
- **Dataexporter** — Plugin complementario para extraer datos de Figma como JSON.

---

## 🔑 Filosofía del dataset

El dataset se construye **sin normalización destructiva**, manteniendo:
- Estructura jerárquica real (`FRAME`, `GROUP`, `VECTOR`, `TEXT`…)
- Propiedades clave: posición (`x`,`y`), dimensiones, texto (`characters`), color.
- Jerarquía `parent → children` para entrenar un **layout coherente**.

---

## 🧩 Estructura de datos para entrenamiento

Cada muestra de entrenamiento es un par `{ prompt, layout }`:
```json
{
  "prompt": "Pantalla de inicio de sesión, campos centrados, input usuario, input contraseña, botón azul grande, fondo claro, diseño minimalista",
  "layout": [ ...estructura JSON exportada por Dataexporter... ]
}
```

El modelo se entrena como **encoder-decoder**, convirtiendo la secuencia de fragmentos en un árbol de nodos.

---

## ✏️ Cómo escribir un buen prompt

**Los prompts se escriben en español**, separados por **comas**, siguiendo una estructura simple:

| Bloque | Ejemplos |
|--------|----------|
| **Contexto** | `Pantalla de inicio de sesión`, `Landing page ecommerce`, `Dashboard web` |
| **Estructura física** | `barra lateral izquierda`, `menú superior`, `tarjetas con estadísticas` |
| **Elementos concretos** | `input usuario`, `botón azul grande`, `gráfico circular` |
| **Estilo visual** | `fondo claro`, `diseño minimalista`, `colores corporativos` |

✅ **Buenas prácticas:**
- Máximo 5–12 fragmentos separados por comas.
- Describir **qué debe existir** y **cómo debe lucir**.
- Usar adjetivos o estilos: `minimalista`, `moderno`, `corporativo`.

**📌 Ejemplos:**

```prompt
Pantalla de registro, formulario centrado, input nombre, input correo, botón verde grande, fondo claro, diseño limpio
```

```
Dashboard de ventas, barra lateral izquierda, menú superior oscuro, tarjetas de métricas, gráfico de líneas, colores corporativos
```

```
Página de producto, imagen destacada grande, galería de imágenes, precio visible, botón comprar azul, sección de reseñas, estilo moderno
```

---

## 🚀 Uso básico

**1️⃣ Exporta datos reales** con **Dataexporter** desde Figma.  
**2️⃣ Asigna un prompt manualmente a cada layout exportado.**  
**3️⃣ Usa `train.py` para entrenar el modelo PyTorch.  
**4️⃣ Exporta a ONNX con `export_onnx.py` y convierte a TF.js con `convert_tfjs.py`.  
**5️⃣ Copia `model.json` y `weights.bin` dentro de `/PLUGIN/model/`.  
**6️⃣ En Figma:**  
- Instala el plugin.
- Escribe el prompt.
- Haz clic en **Generar**.
- ¡El wireframe aparece automáticamente!

---

## ⚡ Visión a futuro

- Mejorar la calidad del dataset con más variantes y ejemplos.
- Probar arquitecturas más robustas para NLP (mini-Transformers entrenados desde cero).
- Publicar dataset anonimizado como recurso abierto.
- Extender el plugin a mobile, tablet y diseños responsivos.
- Crear un modo **pedagógico** que muestre **cómo se interpreta cada fragmento del prompt**.

---

## 📚 Créditos

**Rebeca Sedai** es un proyecto de **IA pedagógica**, desarrollado como ejercicio de tesis académica para explorar cómo el lenguaje natural puede democratizar la creación de interfaces gráficas.  
Diseñado, documentado y entrenado **sin depender de APIs externas**.

---

## 🫶 Contribuir

Este proyecto está pensado para servir como base para estudiantes y curiosos del **NLP aplicado al diseño**.  
Pull requests, issues y forks son bienvenidos siempre que respeten el objetivo de mantener la IA **entrenada localmente**, sin dependencias externas.

---

**🧙‍♀️ Rebeca Sedai — Donde las palabras se convierten en interfaces.**
