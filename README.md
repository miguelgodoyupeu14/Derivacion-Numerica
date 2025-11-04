# 🧮 Calculadora de Derivadas Numéricas

Una aplicación web robusta desarrollada en Python con Flask para resolver problemas de derivadas numéricas usando múltiples métodos matemáticos avanzados.

## 📋 Características

### ✨ Métodos Implementados

1. **Diferencias Finitas**
   - Diferencias hacia adelante, hacia atrás y centradas
   - Soporte para primera y segunda derivada
   - Comparación automática con derivada exacta
   - Validaciones robustas de parámetros

2. **Extrapolación de Richardson**
   - Mejora sistemática de la precisión
   - Múltiples niveles de extrapolación (3-5)
   - Tabla completa de Richardson
   - Análisis de mejora en precisión

3. **Interpolación Polinómica**
   - Método de Lagrange con pasos detallados
   - Método de Newton con tabla de diferencias divididas
   - Evaluación en puntos específicos
   - Construcción del polinomio completo

4. **Derivadas por Interpolación**
   - Combina interpolación con derivación analítica
   - Número configurable de puntos
   - Soporte para ambos métodos (Lagrange/Newton)
   - Mayor precisión que diferencias finitas simples

### 🔧 Características Técnicas

- **Validaciones Robustas**: Verificación exhaustiva de parámetros de entrada
- **Manejo de Errores**: Mensajes informativos y recuperación de errores
- **Interfaz Intuitiva**: Web responsive con pestañas organizadas
- **Cálculos Precisos**: Uso de SymPy para cálculos simbólicos exactos
- **Comparaciones**: Análisis automático de errores vs derivadas exactas

## 🚀 Instalación y Uso

### 1. Requisitos Previos

```bash
Python 3.7 o superior
```

### 2. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5001`

## 📖 Guía de Uso

### Diferencias Finitas

1. Ingresa la función matemática (ej: `x**2 + 2*x + 1`)
2. Especifica el punto donde calcular la derivada
3. Ajusta el paso `h` (recomendado: 1e-5)
4. Selecciona el método (centrada es más precisa)
5. Elige el orden de la derivada (1ª o 2ª)

**Ejemplo:**
- Función: `sin(x)`
- Punto: `1.5708` (π/2)
- Método: Centrada
- Resultado esperado: cos(π/2) ≈ 0

### Extrapolación de Richardson

1. Define la función a derivar
2. Establece el punto de evaluación
3. Configura el paso inicial (recomendado: 1e-2)
4. Selecciona el método base
5. Elige el número de niveles de extrapolación

**Ventajas:**
- Reduce el error de truncamiento
- Mejora automática de precisión
- Visualización completa del proceso

### Interpolación Polinómica

1. Agrega los puntos (x, y) conocidos
2. Selecciona el método (Lagrange o Newton)
3. Especifica el punto donde evaluar
4. Obtén el valor interpolado y el polinomio completo

**Aplicaciones:**
- Aproximación de funciones
- Análisis de datos experimentales
- Predicción de valores intermedios

### Derivadas por Interpolación

1. Ingresa la función continua
2. Define el punto de interés
3. Ajusta el espaciado entre puntos
4. Selecciona el número de puntos (más puntos = mayor precisión)
5. Elige el orden de la derivada

**Ventajas sobre diferencias finitas:**
- Mayor precisión para funciones suaves
- Mejor comportamiento numérico
- Aprovecha la continuidad de la función

## 🧪 Ejemplos de Funciones

### Funciones Polinómicas
```
x**2 + 3*x + 1
x**3 - 2*x**2 + x - 5
```

### Funciones Trigonométricas
```
sin(x)
cos(x)
tan(x)
```

### Funciones Exponenciales y Logarítmicas
```
exp(x)
log(x)
x * exp(-x)
```

### Funciones Compuestas
```
sin(x**2)
exp(-x**2)
x**2 * cos(x)
```

## 📊 Interpretación de Resultados

### Error Absoluto
```
Error = |Derivada_Numérica - Derivada_Exacta|
```

### Niveles de Precisión
- **Excelente**: Error < 1e-10
- **Buena**: Error < 1e-6  
- **Aceptable**: Error < 1e-3
- **Mejorable**: Error > 1e-3

### Recomendaciones

1. **Para funciones suaves**: Usa diferencias centradas o Richardson
2. **Para datos discretos**: Usa interpolación polinómica
3. **Para máxima precisión**: Combina Richardson con diferencias centradas
4. **Para funciones oscilatorias**: Reduce el paso h gradualmente

## 🔍 Solución de Problemas

### Error: "División por cero"
- **Causa**: Puntos x repetidos en interpolación
- **Solución**: Verifica que todos los puntos x sean únicos

### Error: "Función no evaluable"
- **Causa**: Función matemática inválida o dominio incorrecto
- **Solución**: Revisa la sintaxis y el dominio de la función

### Baja precisión en resultados
- **Causas**: Paso h inadecuado, función discontinua
- **Soluciones**: Ajusta h, usa Richardson, verifica continuidad

### Error: "Valores no finitos"
- **Causa**: Función produce infinitos o NaN
- **Solución**: Cambia el punto de evaluación o la función

## 🏗️ Arquitectura del Código

```
DerivadasN/
├── app.py                 # Aplicación principal Flask
├── templates/
│   └── index.html        # Interfaz web responsive
├── requirements.txt      # Dependencias Python
└── README.md            # Documentación
```

### Estructura de app.py
- **Diferencias Finitas**: Implementación de los 3 métodos
- **Richardson**: Extrapolación multinivel
- **Interpolación**: Lagrange y Newton completos
- **API REST**: Endpoints JSON para cada método
- **Validaciones**: Verificaciones exhaustivas

## 🔬 Fundamentos Matemáticos

### Diferencias Finitas

**Hacia Adelante:**
```
f'(x) ≈ [f(x+h) - f(x)] / h
```

**Hacia Atrás:**
```
f'(x) ≈ [f(x) - f(x-h)] / h
```

**Centrada:**
```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```

### Richardson
```
R(n,m) = [4^m * R(n+1,m-1) - R(n,m-1)] / (4^m - 1)
```

### Lagrange
```
P(x) = Σ yi * Li(x)
Li(x) = Π (x - xj) / (xi - xj)  para j≠i
```

### Newton
```
P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ...
```

## 📄 Licencia

Este proyecto está desarrollado con fines educativos y de investigación.

## 👤 Autor

Desarrollado como herramienta robusta para cálculo numérico avanzado.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

**¡Disfruta calculando derivadas numéricas con precisión! 🧮✨**