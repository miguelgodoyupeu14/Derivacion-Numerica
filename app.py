from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
from sympy import symbols, expand, simplify, nsimplify, diff, lambdify
import traceback
import math

app = Flask(__name__)

# ================ DIFERENCIAS FINITAS ================

def diferencias_finitas(f_expr, x_val, h=1e-5, metodo='centrada', orden=1):
    """
    Calcula derivadas numéricas usando diferencias finitas.
    
    Args:
        f_expr: Expresión de la función como string
        x_val: Punto donde evaluar la derivada
        h: Paso para la diferencia finita
        metodo: 'adelante', 'atras', 'centrada'
        orden: Orden de la derivada (1 o 2)
    """
    try:
        # Crear función evaluable
        x = symbols('x')
        expr = sp.sympify(f_expr)
        f = lambdify(x, expr, 'numpy')
        
        # Validaciones
        if abs(h) < 1e-15:
            raise ValueError("h debe ser mayor que 1e-15")
        
        if not np.isfinite(x_val):
            raise ValueError("x debe ser un número finito")
            
        # Verificar que la función se puede evaluar
        test_vals = [x_val, x_val + h, x_val - h]
        for val in test_vals:
            try:
                result = f(val)
                if not np.isfinite(result):
                    raise ValueError(f"La función produce valores no finitos en x={val}")
            except:
                raise ValueError(f"La función no se puede evaluar en x={val}")
        
        pasos = []
        resultado = None
        
        if metodo == 'adelante':
            if orden == 1:
                pasos.append(f"Diferencia hacia adelante de orden 1:")
                pasos.append(f"f'(x) ≈ [f(x+h) - f(x)] / h")
                pasos.append(f"f'({x_val}) ≈ [f({x_val}+{h}) - f({x_val})] / {h}")
                f_x_h = f(x_val + h)
                f_x = f(x_val)
                pasos.append(f"f'({x_val}) ≈ [{f_x_h:.8f} - {f_x:.8f}] / {h}")
                resultado = (f_x_h - f_x) / h
                pasos.append(f"f'({x_val}) ≈ {resultado:.8f}")
            else:
                pasos.append(f"Diferencia hacia adelante de orden 2:")
                pasos.append(f"f''(x) ≈ [f(x+2h) - 2f(x+h) + f(x)] / h²")
                f_x_2h = f(x_val + 2*h)
                f_x_h = f(x_val + h)
                f_x = f(x_val)
                pasos.append(f"f''({x_val}) ≈ [{f_x_2h:.8f} - 2({f_x_h:.8f}) + {f_x:.8f}] / {h**2}")
                resultado = (f_x_2h - 2*f_x_h + f_x) / (h**2)
                pasos.append(f"f''({x_val}) ≈ {resultado:.8f}")
                
        elif metodo == 'atras':
            if orden == 1:
                pasos.append(f"Diferencia hacia atrás de orden 1:")
                pasos.append(f"f'(x) ≈ [f(x) - f(x-h)] / h")
                f_x = f(x_val)
                f_x_h = f(x_val - h)
                pasos.append(f"f'({x_val}) ≈ [{f_x:.8f} - {f_x_h:.8f}] / {h}")
                resultado = (f_x - f_x_h) / h
                pasos.append(f"f'({x_val}) ≈ {resultado:.8f}")
            else:
                pasos.append(f"Diferencia hacia atrás de orden 2:")
                pasos.append(f"f''(x) ≈ [f(x) - 2f(x-h) + f(x-2h)] / h²")
                f_x = f(x_val)
                f_x_h = f(x_val - h)
                f_x_2h = f(x_val - 2*h)
                pasos.append(f"f''({x_val}) ≈ [{f_x:.8f} - 2({f_x_h:.8f}) + {f_x_2h:.8f}] / {h**2}")
                resultado = (f_x - 2*f_x_h + f_x_2h) / (h**2)
                pasos.append(f"f''({x_val}) ≈ {resultado:.8f}")
                
        elif metodo == 'centrada':
            if orden == 1:
                pasos.append(f"Diferencia centrada de orden 1:")
                pasos.append(f"f'(x) ≈ [f(x+h) - f(x-h)] / (2h)")
                f_x_h_pos = f(x_val + h)
                f_x_h_neg = f(x_val - h)
                pasos.append(f"f'({x_val}) ≈ [{f_x_h_pos:.8f} - {f_x_h_neg:.8f}] / {2*h}")
                resultado = (f_x_h_pos - f_x_h_neg) / (2 * h)
                pasos.append(f"f'({x_val}) ≈ {resultado:.8f}")
            else:
                pasos.append(f"Diferencia centrada de orden 2:")
                pasos.append(f"f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²")
                f_x_h_pos = f(x_val + h)
                f_x = f(x_val)
                f_x_h_neg = f(x_val - h)
                pasos.append(f"f''({x_val}) ≈ [{f_x_h_pos:.8f} - 2({f_x:.8f}) + {f_x_h_neg:.8f}] / {h**2}")
                resultado = (f_x_h_pos - 2*f_x + f_x_h_neg) / (h**2)
                pasos.append(f"f''({x_val}) ≈ {resultado:.8f}")
        
        # Calcular derivada exacta para comparación
        try:
            derivada_expr = diff(expr, x, orden)
            derivada_exacta = float(derivada_expr.subs(x, x_val))
            error = abs(resultado - derivada_exacta)
            pasos.append(f"\n<b>Comparación con derivada exacta:</b>")
            pasos.append(f"Derivada exacta: {derivada_exacta:.8f}")
            pasos.append(f"Error absoluto: {error:.2e}")
        except:
            pasos.append("\nNo se pudo calcular la derivada exacta")
            
        return resultado, pasos
        
    except Exception as e:
        return None, [f"Error: {str(e)}"]

# ================ EXTRAPOLACIÓN DE RICHARDSON ================

def richardson_extrapolacion(f_expr, x_val, h_inicial=1e-2, metodo='centrada', orden=1, niveles=3):
    """
    Mejora la precisión usando extrapolación de Richardson.
    """
    try:
        x = symbols('x')
        expr = sp.sympify(f_expr)
        f = lambdify(x, expr, 'numpy')
        
        pasos = []
        pasos.append(f"<b>Extrapolación de Richardson - {metodo.title()}</b>")
        pasos.append(f"Método: {metodo}, Orden: {orden}")
        pasos.append(f"h inicial: {h_inicial}")
        
        # Tabla de Richardson
        R = np.zeros((niveles, niveles))
        h_values = []
        
        # Primera columna: aproximaciones con diferentes h
        for i in range(niveles):
            h = h_inicial / (2**i)
            h_values.append(h)
            
            if metodo == 'centrada' and orden == 1:
                R[i, 0] = (f(x_val + h) - f(x_val - h)) / (2 * h)
            elif metodo == 'adelante' and orden == 1:
                R[i, 0] = (f(x_val + h) - f(x_val)) / h
            elif metodo == 'atras' and orden == 1:
                R[i, 0] = (f(x_val) - f(x_val - h)) / h
            elif metodo == 'centrada' and orden == 2:
                R[i, 0] = (f(x_val + h) - 2*f(x_val) + f(x_val - h)) / (h**2)
            
            pasos.append(f"R[{i},0] con h={h:.6f}: {R[i, 0]:.8f}")
        
        # Extrapolación de Richardson
        pasos.append(f"\n<b>Tabla de extrapolación:</b>")
        for j in range(1, niveles):
            for i in range(niveles - j):
                # Factor 4 para orden 2, factor 2^(2*j) para orden general
                factor = 4**j if orden == 1 else (2**(orden*j + j))
                R[i, j] = (factor * R[i+1, j-1] - R[i, j-1]) / (factor - 1)
                pasos.append(f"R[{i},{j}] = ({factor} * R[{i+1},{j-1}] - R[{i},{j-1}]) / {factor-1} = {R[i, j]:.8f}")
        
        # El mejor resultado está en R[0, niveles-1]
        mejor_resultado = R[0, niveles-1]
        
        # Tabla HTML para mejor visualización
        tabla_html = "<table border='1' style='border-collapse:collapse;margin:10px 0;'>"
        tabla_html += "<tr><th>i</th><th>h</th>"
        for j in range(niveles):
            tabla_html += f"<th>R[i,{j}]</th>"
        tabla_html += "</tr>"
        
        for i in range(niveles):
            tabla_html += f"<tr><td>{i}</td><td>{h_values[i]:.6f}</td>"
            for j in range(niveles):
                if j <= niveles - 1 - i:
                    tabla_html += f"<td>{R[i, j]:.8f}</td>"
                else:
                    tabla_html += "<td>-</td>"
            tabla_html += "</tr>"
        tabla_html += "</table>"
        
        pasos.append(f"\n{tabla_html}")
        pasos.append(f"\n<b>Resultado mejorado:</b> {mejor_resultado:.10f}")
        
        # Comparar con derivada exacta
        try:
            derivada_expr = diff(expr, x, orden)
            derivada_exacta = float(derivada_expr.subs(x, x_val))
            error_inicial = abs(R[0, 0] - derivada_exacta)
            error_mejorado = abs(mejor_resultado - derivada_exacta)
            
            pasos.append(f"\n<b>Comparación:</b>")
            pasos.append(f"Derivada exacta: {derivada_exacta:.10f}")
            pasos.append(f"Error inicial: {error_inicial:.2e}")
            pasos.append(f"Error mejorado: {error_mejorado:.2e}")
            pasos.append(f"Mejora en precisión: {error_inicial/error_mejorado:.2f}x")
        except:
            pass
        
        return mejor_resultado, pasos
        
    except Exception as e:
        return None, [f"Error en Richardson: {str(e)}"]

# ================ INTERPOLACIÓN POLINÓMICA ================

def lagrange_interpolacion(x_eval, xs, ys):
    """Interpolación de Lagrange con pasos detallados"""
    n = len(xs)
    resultado = 0
    pasos = []
    x_sym = symbols('x')
    
    pasos.append(f"<b>Interpolación de Lagrange</b>")
    pasos.append(f"Puntos: {list(zip(xs, ys))}")
    
    for i in range(n):
        # Construir L_i(x) simbólico
        Li = 1
        Li_expr = "1"
        for j in range(n):
            if i != j:
                Li *= (x_sym - xs[j]) / (xs[i] - xs[j])
                Li_expr += f" * (x - {xs[j]}) / ({xs[i]} - {xs[j]})"
        
        Li_simpl = nsimplify(simplify(expand(Li)), rational=True)
        Li_val = float(Li.subs(x_sym, x_eval))
        termino_val = ys[i] * Li_val
        resultado += termino_val
        
        pasos.append(f"L_{i}(x) = {Li_simpl}")
        pasos.append(f"L_{i}({x_eval}) = {Li_val:.6f}")
        pasos.append(f"Término {i}: {ys[i]} * {Li_val:.6f} = {termino_val:.6f}")
    
    pasos.append(f"\n<b>Resultado:</b> P({x_eval}) = {resultado:.6f}")
    return resultado, pasos

def lagrange_polinomio(xs, ys):
    """Obtiene el polinomio de Lagrange expandido"""
    x = symbols('x')
    n = len(xs)
    polinomio = 0
    pasos = []
    
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - xs[j])/(xs[i] - xs[j])
        polinomio += ys[i] * Li
        pasos.append(f"Término {i}: {ys[i]} * L_{i}(x)")
    
    polinomio_expandido = simplify(expand(polinomio))
    pasos.append(f"Polinomio expandido: {polinomio_expandido}")
    return polinomio_expandido, pasos

def newton_diferencias_divididas(xs, ys, x_eval=None):
    """Interpolación de Newton con diferencias divididas"""
    n = len(xs)
    if len(set(xs)) != len(xs):
        raise Exception("Hay valores de x repetidos")
    
    # Tabla de diferencias divididas
    dd = [ys[:]]
    pasos = []
    
    pasos.append(f"<b>Interpolación de Newton</b>")
    pasos.append(f"Puntos: {list(zip(xs, ys))}")
    
    # Calcular diferencias divididas
    for j in range(1, n):
        col = []
        for i in range(n-j):
            divisor = xs[i+j] - xs[i]
            if divisor == 0:
                raise Exception(f"División por cero: x[{i+j}] = x[{i}]")
            val = (dd[j-1][i+1] - dd[j-1][i]) / divisor
            col.append(val)
        dd.append(col)
    
    # Tabla HTML
    tabla_html = "<table border='1' style='border-collapse:collapse;margin:10px 0;'>"
    tabla_html += "<tr><th>i</th><th>x_i</th><th>f[x_i]</th>"
    for j in range(1, n):
        tabla_html += f"<th>Orden {j}</th>"
    tabla_html += "</tr>"
    
    for i in range(n):
        tabla_html += f"<tr><td>{i}</td><td>{xs[i]}</td><td>{ys[i]}</td>"
        for j in range(1, n):
            if i + j < n:
                idx = i
                if j-1 < len(dd) and idx < len(dd[j]):
                    tabla_html += f"<td>{dd[j][idx]:.6f}</td>"
                else:
                    tabla_html += "<td>-</td>"
            else:
                tabla_html += "<td>-</td>"
        tabla_html += "</tr>"
    
    tabla_html += "</table>"
    pasos.append(tabla_html)
    
    # Coeficientes
    coef = [dd[j][0] for j in range(n)]
    pasos.append(f"Coeficientes: {[f'{c:.6f}' for c in coef]}")
    
    if x_eval is not None:
        # Evaluar polinomio
        resultado = coef[0]
        mult = 1
        pasos.append(f"\nEvaluación en x = {x_eval}:")
        pasos.append(f"P({x_eval}) = {coef[0]:.6f}")
        
        for i in range(1, n):
            mult *= (x_eval - xs[i-1])
            termino = coef[i] * mult
            resultado += termino
            pasos.append(f"+ {coef[i]:.6f} * {mult:.6f} = {termino:.6f}")
        
        pasos.append(f"\n<b>Resultado:</b> P({x_eval}) = {resultado:.6f}")
        return resultado, pasos
    
    return None, pasos

def derivada_por_interpolacion(f_expr, x_val, h=1e-3, num_puntos=5, metodo='lagrange', orden=1):
    """Calcula derivadas usando interpolación polinómica"""
    try:
        x = symbols('x')
        expr = sp.sympify(f_expr)
        f = lambdify(x, expr, 'numpy')
        
        if num_puntos < orden + 1:
            raise ValueError(f"Se necesitan al menos {orden + 1} puntos")
        
        if num_puntos % 2 == 0:
            num_puntos += 1
        
        # Generar puntos centrados
        inicio = -(num_puntos // 2)
        fin = num_puntos // 2 + 1
        x_puntos = [x_val + i * h for i in range(inicio, fin)]
        y_puntos = [f(xi) for xi in x_puntos]
        
        pasos = []
        pasos.append(f"<b>Derivada por interpolación {metodo.title()}</b>")
        pasos.append(f"Puntos utilizados: {list(zip(x_puntos, y_puntos))}")
        
        # Obtener polinomio
        if metodo == 'lagrange':
            polinomio, pasos_poli = lagrange_polinomio(x_puntos, y_puntos)
        else:  # newton
            # Para Newton necesitamos construir el polinomio simbólicamente
            coef_newton = []
            dd = [y_puntos[:]]
            for j in range(1, len(x_puntos)):
                col = []
                for i in range(len(x_puntos)-j):
                    val = (dd[j-1][i+1] - dd[j-1][i]) / (x_puntos[i+j] - x_puntos[i])
                    col.append(val)
                dd.append(col)
            
            polinomio = dd[0][0]
            mult = 1
            for i in range(1, len(x_puntos)):
                mult *= (x - x_puntos[i-1])
                polinomio += dd[i][0] * mult
            
            polinomio = simplify(expand(polinomio))
        
        # Derivar el polinomio
        derivada_poly = diff(polinomio, x, orden)
        resultado = float(derivada_poly.subs(x, x_val))
        
        pasos.append(f"Polinomio: {polinomio}")
        pasos.append(f"Derivada de orden {orden}: {derivada_poly}")
        pasos.append(f"Evaluado en x={x_val}: {resultado:.8f}")
        
        # Comparar con derivada exacta
        try:
            derivada_exacta_expr = diff(expr, x, orden)
            derivada_exacta = float(derivada_exacta_expr.subs(x, x_val))
            error = abs(resultado - derivada_exacta)
            pasos.append(f"\nDerivada exacta: {derivada_exacta:.8f}")
            pasos.append(f"Error: {error:.2e}")
        except:
            pass
        
        return resultado, pasos
        
    except Exception as e:
        return None, [f"Error: {str(e)}"]

# ================ RUTAS DE FLASK ================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diferencias_finitas', methods=['POST'])
def calcular_diferencias():
    try:
        data = request.get_json()
        f_expr = data['funcion']
        x_val = float(data['x'])
        h = float(data.get('h', 1e-5))
        metodo = data['metodo']
        orden = int(data.get('orden', 1))
        
        resultado, pasos = diferencias_finitas(f_expr, x_val, h, metodo, orden)
        
        return jsonify({
            'success': True,
            'resultado': resultado,
            'pasos': pasos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/richardson', methods=['POST'])
def calcular_richardson():
    try:
        data = request.get_json()
        f_expr = data['funcion']
        x_val = float(data['x'])
        h = float(data.get('h', 1e-2))
        metodo = data['metodo']
        orden = int(data.get('orden', 1))
        niveles = int(data.get('niveles', 3))
        
        resultado, pasos = richardson_extrapolacion(f_expr, x_val, h, metodo, orden, niveles)
        
        return jsonify({
            'success': True,
            'resultado': resultado,
            'pasos': pasos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/interpolacion', methods=['POST'])
def calcular_interpolacion():
    try:
        data = request.get_json()
        metodo = data['metodo']
        x_puntos = [float(x) for x in data['x_puntos']]
        y_puntos = [float(y) for y in data['y_puntos']]
        x_eval = float(data.get('x_eval', 0))
        
        if len(x_puntos) != len(y_puntos):
            raise ValueError("Número diferente de puntos x e y")
        
        if len(set(x_puntos)) != len(x_puntos):
            raise ValueError("Puntos x repetidos")
        
        if metodo == 'lagrange':
            resultado, pasos = lagrange_interpolacion(x_eval, x_puntos, y_puntos)
        else:  # newton
            resultado, pasos = newton_diferencias_divididas(x_puntos, y_puntos, x_eval)
        
        return jsonify({
            'success': True,
            'resultado': resultado,
            'pasos': pasos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/derivada_interpolacion', methods=['POST'])
def calcular_derivada_interpolacion():
    try:
        data = request.get_json()
        f_expr = data['funcion']
        x_val = float(data['x'])
        h = float(data.get('h', 1e-3))
        num_puntos = int(data.get('num_puntos', 5))
        metodo = data['metodo']
        orden = int(data.get('orden', 1))
        
        resultado, pasos = derivada_por_interpolacion(f_expr, x_val, h, num_puntos, metodo, orden)
        
        return jsonify({
            'success': True,
            'resultado': resultado,
            'pasos': pasos
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)