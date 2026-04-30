"""
╔══════════════════════════════════════════════════════════════════╗
║     ANALIZADOR DE POSTURA EN EJERCICIO FÍSICO                   ║
║     YOLOv8-Pose + yt_dlp + OpenCV                               ║
╚══════════════════════════════════════════════════════════════════╝

Sin selección de ROI — análisis directo sobre el frame completo.
Sin límite de tiempo — corre hasta fin de vídeo o tecla Q.

Vista compatible:
  - De LADO   → usa el keypoint del lado con mayor confianza
  - De FRENTE → promedia keypoints izquierdo + derecho (más robusto)
    La detección de orientación es automática en cada frame.

Contador de repeticiones compacto dentro del panel HUD.

Instalación:
    pip install ultralytics opencv-python numpy yt-dlp

Uso:
    python pose_analyzer_youtube.py                           # URL por defecto
    python pose_analyzer_youtube.py --url "URL_DE_YOUTUBE"   # vídeo personalizado
    python pose_analyzer_youtube.py --source 0               # webcam local
    python pose_analyzer_youtube.py --exercise deadlift      # otro ejercicio

Ejercicios soportados:
    squat    → Sentadilla
    deadlift → Peso muerto
    pushup   → Flexión de brazos
"""

import cv2
import yt_dlp
import numpy as np
import argparse
import time
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ═══════════════════════════════════════════════════════════════════

DEFAULT_URL = "https://www.youtube.com/watch?v=aclHkVaku9U"
MODEL_SIZE  = "n"
CONF        = 0.50

# ═══════════════════════════════════════════════════════════════════
#  ÍNDICES DE KEYPOINTS — COCO 17 puntos
# ═══════════════════════════════════════════════════════════════════
KP = {
    "nose":           0,
    "left_eye":       1,  "right_eye":      2,
    "left_ear":       3,  "right_ear":      4,
    "left_shoulder":  5,  "right_shoulder": 6,
    "left_elbow":     7,  "right_elbow":    8,
    "left_wrist":     9,  "right_wrist":    10,
    "left_hip":       11, "right_hip":      12,
    "left_knee":      13, "right_knee":     14,
    "left_ankle":     15, "right_ankle":    16,
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ═══════════════════════════════════════════════════════════════════
#  PALETA DE COLORES (BGR)
# ═══════════════════════════════════════════════════════════════════
C = {
    "ok":      (78,  220,   0),
    "warning": (20,  176, 255),
    "error":   (30,   30, 240),
    "accent":  (255, 220, 100),
    "white":   (255, 255, 255),
    "panel":   (25,   25,  40),
    "muted":   (120, 130, 160),
    "black":   (0,     0,   0),
}

# ═══════════════════════════════════════════════════════════════════
#  FUNCIONES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════

def angle_3pts(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc  = a - b, c - b
    cos_v   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def get_kp(keypoints, idx, conf_thresh=0.30):
    if keypoints is None or idx >= len(keypoints):
        return None
    x, y, conf = float(keypoints[idx][0]), float(keypoints[idx][1]), float(keypoints[idx][2])
    return (x, y) if conf >= conf_thresh else None

def get_conf(keypoints, idx):
    if keypoints is None or idx >= len(keypoints):
        return 0.0
    return float(keypoints[idx][2])

def max_status(current, new):
    order = {"ok": 0, "warning": 1, "error": 2}
    return new if order.get(new, 0) > order.get(current, 0) else current

# ═══════════════════════════════════════════════════════════════════
#  DETECCIÓN DE ORIENTACIÓN Y RESOLUCIÓN DE KEYPOINTS BILATERALES
# ═══════════════════════════════════════════════════════════════════

def detect_orientation(kp_raw):
    """
    Criterio geométrico basado en la observación real del movimiento:

    FRENTE:
      - Ambos hombros visibles y SEPARADOS horizontalmente (persona de cara)
      - La distancia cadera-rodilla de AMBOS lados es similar entre sí
        (al hacer sentadilla de frente, los dos lados bajan igual)

    LADO:
      - Los hombros están muy juntos horizontalmente (uno tapa al otro)
        O solo uno es visible con buena confianza

    Señal principal  → separación horizontal de hombros
    Señal secundaria → simetría entre cadera-rodilla izquierda y derecha
                       (refuerza la detección cuando los hombros son ambiguos)

    Retorna: "front" | "side"
    """
    ls = get_kp(kp_raw, KP["left_shoulder"],  conf_thresh=0.35)
    rs = get_kp(kp_raw, KP["right_shoulder"], conf_thresh=0.35)
    lc = get_conf(kp_raw, KP["left_shoulder"])
    rc = get_conf(kp_raw, KP["right_shoulder"])

    # ── Señal 1: separación horizontal de hombros ──
    # De frente los hombros están separados; de lado uno tapa al otro
    shoulder_sep = abs(ls[0] - rs[0]) if (ls and rs) else 0.0

    # Necesitamos una referencia de escala: usamos la altura del torso
    # (distancia vertical hombro-cadera) para normalizar la separación
    lh = get_kp(kp_raw, KP["left_hip"],  conf_thresh=0.35)
    rh = get_kp(kp_raw, KP["right_hip"], conf_thresh=0.35)

    torso_height = 1.0
    if ls and lh:
        torso_height = max(abs(ls[1] - lh[1]), 1.0)
    elif rs and rh:
        torso_height = max(abs(rs[1] - rh[1]), 1.0)

    # Ratio separación_hombros / altura_torso
    # De frente suele ser > 0.5; de lado suele ser < 0.25
    shoulder_ratio = shoulder_sep / torso_height

    # ── Señal 2: simetría de distancias cadera-rodilla ──
    # De frente ambas piernas bajan juntas → distancias similares
    # De lado una pierna tapa a la otra → asimetría o una no visible
    lk = get_kp(kp_raw, KP["left_knee"],  conf_thresh=0.35)
    rk = get_kp(kp_raw, KP["right_knee"], conf_thresh=0.35)

    hip_knee_symmetric = False
    if lh and rh and lk and rk:
        dist_l = np.hypot(lh[0]-lk[0], lh[1]-lk[1])
        dist_r = np.hypot(rh[0]-rk[0], rh[1]-rk[1])
        # Simetría: diferencia relativa entre ambas distancias
        diff_ratio = abs(dist_l - dist_r) / (max(dist_l, dist_r) + 1e-8)
        hip_knee_symmetric = diff_ratio < 0.25   # menos del 25% de diferencia

    # ── Decisión final ──
    # Frente clara: hombros muy separados (ratio > 0.50)
    if shoulder_ratio > 0.50 and lc > 0.35 and rc > 0.35:
        return "front"

    # Frente probable: hombros moderadamente separados + piernas simétricas
    if shoulder_ratio > 0.30 and hip_knee_symmetric and lc > 0.30 and rc > 0.30:
        return "front"

    # Lado: hombros juntos o solo uno visible
    return "side"

def bilateral_kp(kp_raw, left_idx, right_idx, conf_thresh=0.30):
    """Frente: promedio de ambos lados si están disponibles."""
    lp = get_kp(kp_raw, left_idx,  conf_thresh)
    rp = get_kp(kp_raw, right_idx, conf_thresh)
    if lp and rp:
        return midpoint(lp, rp)
    return lp or rp

def best_side_kp(kp_raw, left_idx, right_idx, conf_thresh=0.30):
    """Lado: elige el keypoint con mayor confianza."""
    lc = get_conf(kp_raw, left_idx)
    rc = get_conf(kp_raw, right_idx)
    if lc >= rc:
        return get_kp(kp_raw, left_idx, conf_thresh) or get_kp(kp_raw, right_idx, conf_thresh)
    return get_kp(kp_raw, right_idx, conf_thresh) or get_kp(kp_raw, left_idx, conf_thresh)

def resolve_kp(kp_raw, left_idx, right_idx, orientation, conf_thresh=0.30):
    if orientation == "front":
        return bilateral_kp(kp_raw, left_idx, right_idx, conf_thresh)
    return best_side_kp(kp_raw, left_idx, right_idx, conf_thresh)

# ═══════════════════════════════════════════════════════════════════
#  ANALIZADORES
# ═══════════════════════════════════════════════════════════════════

class SquatAnalyzer:
    DOWN_THRESH = 120
    UP_THRESH   = 160

    def __init__(self):
        self.reps  = 0
        self.phase = "inicio"

    def analyze(self, kp_raw, orientation="side"):
        feedback, angles, status = [], {}, "ok"

        shoulder = resolve_kp(kp_raw, KP["left_shoulder"], KP["right_shoulder"], orientation)
        hip      = resolve_kp(kp_raw, KP["left_hip"],      KP["right_hip"],      orientation)
        knee     = resolve_kp(kp_raw, KP["left_knee"],     KP["right_knee"],     orientation)
        ankle    = resolve_kp(kp_raw, KP["left_ankle"],    KP["right_ankle"],    orientation)

        # Angulo de rodilla + conteo
        if hip and knee and ankle:
            knee_ang = angle_3pts(hip, knee, ankle)
            angles["Rodilla"] = knee_ang
            if self.phase == "inicio" and knee_ang < self.DOWN_THRESH:
                self.phase = "abajo"
            elif self.phase == "abajo" and knee_ang > self.UP_THRESH:
                self.phase = "inicio"
                self.reps += 1
            if knee_ang > 155:
                feedback.append(("Baja mas la cadera", C["warning"]))
            elif 82 <= knee_ang <= 105:
                feedback.append(("Profundidad ideal!", C["ok"]))
            elif knee_ang < 68:
                feedback.append(("Demasiado profundo", C["warning"]))
            else:
                feedback.append(("Bajando bien", C["ok"]))

        # Inclinacion del torso
        if shoulder and hip:
            dx = shoulder[0] - hip[0]
            dy = shoulder[1] - hip[1]
            torso_ang = abs(float(np.degrees(np.arctan2(dx, -dy + 1e-8))))
            angles["Torso"] = torso_ang
            if torso_ang > 50:
                feedback.append(("Manten la espalda recta!", C["error"]))
                status = "error"
            elif torso_ang > 33:
                feedback.append(("Espalda inclinada", C["warning"]))
                status = max_status(status, "warning")
            else:
                feedback.append(("Espalda recta", C["ok"]))

        # Rodilla no sobrepasa tobillo
        if knee and ankle and abs(knee[0] - ankle[0]) > 45:
            feedback.append(("Rodillas no sobrepasen pies", C["error"]))
            status = "error"

        if not feedback:
            feedback.append(("Postura correcta", C["ok"]))

        return {"feedback": feedback, "angles": angles,
                "status": status, "reps": self.reps, "phase": self.phase,
                "orientation": orientation}


class DeadliftAnalyzer:
    DOWN_THRESH = 120
    UP_THRESH   = 160

    def __init__(self):
        self.reps  = 0
        self.phase = "inicio"

    def analyze(self, kp_raw, orientation="side"):
        feedback, angles, status = [], {}, "ok"

        shoulder = resolve_kp(kp_raw, KP["left_shoulder"], KP["right_shoulder"], orientation)
        hip      = resolve_kp(kp_raw, KP["left_hip"],      KP["right_hip"],      orientation)
        knee     = resolve_kp(kp_raw, KP["left_knee"],     KP["right_knee"],     orientation)
        wrist    = resolve_kp(kp_raw, KP["left_wrist"],    KP["right_wrist"],    orientation)

        # Angulo de cadera + conteo
        if shoulder and hip and knee:
            hip_ang = angle_3pts(shoulder, hip, knee)
            angles["Cadera"] = hip_ang
            if self.phase == "inicio" and hip_ang < self.DOWN_THRESH:
                self.phase = "abajo"
            elif self.phase == "abajo" and hip_ang > self.UP_THRESH:
                self.phase = "inicio"
                self.reps += 1
            if hip_ang > 165:
                feedback.append(("Posicion superior", C["ok"]))
            elif 80 <= hip_ang <= 120:
                feedback.append(("Bisagra de cadera correcta", C["ok"]))
            else:
                feedback.append(("Ajusta el angulo de cadera", C["warning"]))

        # ── Espalda neutra: ángulo del segmento hombro-cadera vs vertical ──
        # Mide cuánto se inclina la espalda hacia adelante.
        # En peso muerto la espalda siempre estará inclinada durante la fase
        # de bajada, lo que importa es que NO se doble (redondee) en exceso.
        #
        # Umbrales empíricos:
        #   < 30°  → casi vertical (posición superior, espalda recta)
        #   30-70° → inclinación normal durante la bajada
        #   > 70°  → espalda muy horizontal, riesgo de redondeo
        #   > 85°  → espalda paralela al suelo o redondeada, peligro
        #
        # Nota: el ángulo sube naturalmente al bajar la barra. Se avisa
        # solo cuando supera el umbral crítico para la fase actual.
        if shoulder and hip:
            dx = shoulder[0] - hip[0]
            dy = shoulder[1] - hip[1]  # negativo porque Y crece hacia abajo
            back_angle = abs(float(np.degrees(np.arctan2(dx, -dy + 1e-8))))
            angles["Espalda"] = back_angle

            # Adaptar el umbral de aviso según la fase del movimiento
            if self.phase == "inicio":
                # Posición superior: espalda debe estar casi vertical
                if back_angle > 30:
                    feedback.append(("Endereza la espalda arriba", C["warning"]))
                    status = max_status(status, "warning")
                else:
                    feedback.append(("Espalda neutra", C["ok"]))
            else:
                # Fase de bajada: algo de inclinación es normal
                if back_angle > 85:
                    feedback.append(("Espalda redondeada - peligro!", C["error"]))
                    status = "error"
                elif back_angle > 70:
                    feedback.append(("Espalda muy inclinada", C["warning"]))
                    status = max_status(status, "warning")
                else:
                    feedback.append(("Espalda neutra", C["ok"]))

        # Barra cerca del cuerpo
        if wrist and hip and abs(wrist[0] - hip[0]) > 90:
            feedback.append(("Barra pegada al cuerpo", C["warning"]))

        if not feedback:
            feedback.append(("Postura correcta", C["ok"]))

        return {"feedback": feedback, "angles": angles,
                "status": status, "reps": self.reps, "phase": self.phase,
                "orientation": orientation}


class PushupAnalyzer:
    DOWN_THRESH = 100
    UP_THRESH   = 155

    def __init__(self):
        self.reps  = 0
        self.phase = "inicio"

    def analyze(self, kp_raw, orientation="side"):
        feedback, angles, status = [], {}, "ok"

        shoulder = resolve_kp(kp_raw, KP["left_shoulder"], KP["right_shoulder"], orientation)
        elbow    = resolve_kp(kp_raw, KP["left_elbow"],    KP["right_elbow"],    orientation)
        wrist    = resolve_kp(kp_raw, KP["left_wrist"],    KP["right_wrist"],    orientation)
        hip      = resolve_kp(kp_raw, KP["left_hip"],      KP["right_hip"],      orientation)
        ankle    = resolve_kp(kp_raw, KP["left_ankle"],    KP["right_ankle"],    orientation)

        # Angulo de codo + conteo
        if shoulder and elbow and wrist:
            elbow_ang = angle_3pts(shoulder, elbow, wrist)
            angles["Codo"] = elbow_ang
            if self.phase == "inicio" and elbow_ang < self.DOWN_THRESH:
                self.phase = "abajo"
            elif self.phase == "abajo" and elbow_ang > self.UP_THRESH:
                self.phase = "inicio"
                self.reps += 1
            if 82 <= elbow_ang <= 100:
                feedback.append(("Profundidad correcta!", C["ok"]))
            elif elbow_ang > 155:
                feedback.append(("Baja mas el cuerpo", C["warning"]))
            else:
                feedback.append(("Bajando bien", C["ok"]))

        # Alineacion corporal
        if shoulder and hip and ankle:
            body_ang = angle_3pts(shoulder, hip, ankle)
            angles["Cuerpo"] = body_ang
            if body_ang < 158:
                feedback.append(("Cadera arriba - bajala", C["error"]))
                status = "error"
            elif body_ang > 188:
                feedback.append(("Cadera hundida - subela", C["error"]))
                status = "error"
            else:
                feedback.append(("Cuerpo alineado", C["ok"]))

        if not feedback:
            feedback.append(("Postura correcta", C["ok"]))

        return {"feedback": feedback, "angles": angles,
                "status": status, "reps": self.reps, "phase": self.phase,
                "orientation": orientation}


EJERCICIOS = {
    "squat":    ("Sentadilla",        SquatAnalyzer),
    "deadlift": ("Peso Muerto",       DeadliftAnalyzer),
    "pushup":   ("Flexion de Brazos", PushupAnalyzer),
}

# ═══════════════════════════════════════════════════════════════════
#  DIBUJO
# ═══════════════════════════════════════════════════════════════════

def draw_skeleton(frame, kp_raw, status):
    color = {"ok": C["ok"], "warning": C["warning"], "error": C["error"]}.get(status, C["ok"])
    for (i, j) in SKELETON:
        p1 = get_kp(kp_raw, i)
        p2 = get_kp(kp_raw, j)
        if p1 and p2:
            cv2.line(frame, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])), color, 2, cv2.LINE_AA)
    for idx in range(17):
        pt = get_kp(kp_raw, idx)
        if pt:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, C["white"], -1)
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, color, 1, cv2.LINE_AA)


def draw_angle_label(frame, pt, angle):
    if pt is None:
        return
    color = C["ok"] if 70 <= angle <= 115 else C["warning"]
    x, y  = int(pt[0]) + 10, int(pt[1]) - 8
    cv2.putText(frame, f"{angle:.0f}g", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C["black"], 3, cv2.LINE_AA)
    cv2.putText(frame, f"{angle:.0f}g", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_hud(frame, result, exercise_name, fps):
    """
    Panel HUD lateral izquierdo.
    Repeticiones compactas: una sola línea — etiqueta + número + fase.
    """
    h, w    = frame.shape[:2]
    panel_w = 252

    # Panel semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), C["panel"], -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.rectangle(frame, (0, 0), (3, h), C["accent"], -1)

    y = 26

    # Título
    cv2.putText(frame, "Daniel Marin Pacheco", (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, C["accent"], 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, exercise_name.upper(), (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C["white"], 2, cv2.LINE_AA)
    y += 5
    cv2.line(frame, (10, y), (panel_w - 10, y), C["accent"], 1)

    # ── Orientación: badge grande visible sobre el vídeo ──
    orient_txt = "FRENTE" if result.get("orientation") == "front" else "LATERAL"
    orient_col = (255, 200, 0) if result.get("orientation") == "front" else (180, 180, 180)
    orient_bg  = (40, 30, 0)   if result.get("orientation") == "front" else (30, 30, 30)

    # Medir ancho del texto para centrar el badge
    (tw, th), _ = cv2.getTextSize(orient_txt, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    bx1 = panel_w + 10
    by1 = 10
    bx2 = bx1 + tw + 20
    by2 = by1 + th + 14

    # Fondo del badge semitransparente
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), orient_bg, -1)
    cv2.addWeighted(badge_overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), orient_col, 1)

    # Icono simple: cuadrado (frente) o línea (lado)
    icon_x, icon_y = bx1 + 8, by1 + th // 2 + 7
    if result.get("orientation") == "front":
        cv2.rectangle(frame, (icon_x - 4, icon_y - 5),
                      (icon_x + 4, icon_y + 5), orient_col, 1)
    else:
        cv2.line(frame, (icon_x, icon_y - 5),
                 (icon_x, icon_y + 5), orient_col, 2)

    cv2.putText(frame, orient_txt, (bx1 + 18, by1 + th + 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, orient_col, 2, cv2.LINE_AA)

    # Etiqueta pequeña en el panel también (referencia rápida)
    y += 14
    cv2.putText(frame, f"Vista: {orient_txt}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, orient_col, 1, cv2.LINE_AA)

    # ── Repeticiones COMPACTAS (una sola línea) ──
    y += 18
    cv2.line(frame, (10, y), (panel_w - 10, y), (55, 55, 75), 1)
    y += 16
    phase_col = C["ok"] if result["phase"] == "inicio" else C["warning"]
    cv2.putText(frame, "REPS:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["muted"], 1, cv2.LINE_AA)
    cv2.putText(frame, str(result["reps"]), (60, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, C["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, result["phase"].upper(), (100, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, phase_col, 1, cv2.LINE_AA)

    # Ángulos con barra de progreso
    y += 18
    cv2.line(frame, (10, y), (panel_w - 10, y), (55, 55, 75), 1)
    y += 14
    cv2.putText(frame, "ANGULOS", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["accent"], 1, cv2.LINE_AA)
    y += 16

    for name, ang in result["angles"].items():
        pct     = min(int((ang / 180) * (panel_w - 22)), panel_w - 22)
        bar_col = C["ok"] if 70 <= ang <= 115 else C["warning"]
        cv2.putText(frame, f"{name}: {ang:.0f}g", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["white"], 1, cv2.LINE_AA)
        y += 13
        cv2.rectangle(frame, (10, y), (panel_w - 10, y + 5), (45, 45, 65), -1)
        cv2.rectangle(frame, (10, y), (10 + pct,     y + 5), bar_col,       -1)
        y += 14

    # Feedback
    y += 4
    cv2.line(frame, (10, y), (panel_w - 10, y), (55, 55, 75), 1)
    y += 14
    cv2.putText(frame, "FEEDBACK", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["accent"], 1, cv2.LINE_AA)
    y += 16

    for msg, col in result["feedback"][:5]:
        cv2.putText(frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        y += 18

    # FPS
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, C["accent"], 1, cv2.LINE_AA)

    # Borde de estado
    border_col = {"ok": C["ok"], "warning": C["warning"], "error": C["error"]}.get(
        result["status"], C["ok"])
    cv2.rectangle(frame, (panel_w + 2, 2), (w - 2, h - 2), border_col, 3)

    # Overlay rojo sutil en error
    if result["status"] == "error":
        flash = frame.copy()
        cv2.rectangle(flash, (panel_w, 0), (w, h), C["error"], -1)
        cv2.addWeighted(flash, 0.07, frame, 0.93, 0, frame)


# ═══════════════════════════════════════════════════════════════════
#  DESCARGA DE VÍDEO con yt_dlp
# ═══════════════════════════════════════════════════════════════════

def download_video(url, out_path="input_video.mp4"):
    """
    Descarga el vídeo de YouTube al disco antes de procesarlo.
    Formato 18 = 360p MP4 (buena compatibilidad con cv2).
    Muestra una barra de progreso en consola.
    """
    def progress_hook(d):
        if d["status"] == "downloading":
            pct = d.get("_percent_str", "?").strip()
            spd = d.get("_speed_str",   "?").strip()
            eta = d.get("_eta_str",     "?").strip()
            print(f"\r  Descargando: {pct:>6}  |  {spd:>10}  |  ETA {eta}   ",
                  end="", flush=True)
        elif d["status"] == "finished":
            print(f"\r  Descarga completa: {d['filename']}          ")

    ydl_opts = {
        "format":           "18/best[ext=mp4]/best",
        "outtmpl":          out_path,
        "quiet":            True,
        "no_warnings":      True,
        "progress_hooks":   [progress_hook],
    }

    print(f"[yt_dlp] Iniciando descarga: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info  = ydl.extract_info(url, download=True)
        title = info.get("title", "Video")

    print(f"[yt_dlp] Guardado como: {out_path}  ({title})\n")
    return out_path


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════

def run(source, exercise="squat", model_size=MODEL_SIZE, conf=CONF):

    exercise_name, AnalyzerClass = EJERCICIOS.get(exercise, EJERCICIOS["squat"])
    analyzer = AnalyzerClass()

    print(f"\n[YOLO] Cargando yolov8{model_size}-pose.pt ...")
    model = YOLO(f"yolov8{model_size}-pose.pt")
    print("[YOLO] Modelo listo.\n")

    # ── Determinar fuente ──
    # Si es URL de YouTube → descargar primero al disco
    if isinstance(source, str) and source.startswith("http"):
        video_path = download_video(source, out_path="input_video.mp4")
        cap = cv2.VideoCapture(video_path)
    elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        # Webcam: captura en tiempo real (sin descarga)
        cap = cv2.VideoCapture(int(source))
        video_path = None
        print(f"[CAM] Camara {source} (modo tiempo real — ventana activa)\n")
    else:
        # Archivo local
        video_path = source
        cap = cv2.VideoCapture(source)
        print(f"[VIDEO] {source}\n")

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la fuente.")
        return

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {frame_w}x{frame_h} @ {fps_src:.0f} fps  |  frames totales: {total_f or '?'}")

    out_path = f"output_{exercise}.mp4"
    out      = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               fps_src, (frame_w, frame_h))
    print(f"[OUT]  Guardando resultado en: {out_path}\n")

    # Webcam necesita mostrar ventana (no hay fichero que guardar previamente)
    use_display = (video_path is None)

    # Mapa: nombre de ángulo → par de índices de keypoint (izq, der)
    ANGLE_KP = {
        "Rodilla": (KP["left_knee"],     KP["right_knee"]),
        "Cadera":  (KP["left_hip"],      KP["right_hip"]),
        "Codo":    (KP["left_elbow"],    KP["right_elbow"]),
        "Espalda": (KP["left_shoulder"], KP["right_shoulder"]),
        "Torso":   (KP["left_shoulder"], KP["right_shoulder"]),
        "Cuerpo":  (KP["left_hip"],      KP["right_hip"]),
    }

    prev_time = time.time()
    fps       = 0.0
    frame_idx = 0
    t_start   = time.time()

    print(f"[OK] Procesando '{exercise_name}' ...")
    if use_display:
        print("     Webcam activa — Q para salir\n")
    else:
        print("     Sin ventana — procesando en segundo plano...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results    = model(frame, conf=conf, verbose=False)
        now        = time.time()
        fps        = 1.0 / (now - prev_time + 1e-9)
        prev_time  = now

        result = {
            "feedback":    [("Sin persona detectada", C["warning"])],
            "angles":      {},
            "status":      "ok",
            "reps":        analyzer.reps,
            "phase":       analyzer.phase,
            "orientation": "side",
        }

        if results and results[0].keypoints is not None:
            kps_all = results[0].keypoints.data
            if len(kps_all) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                best  = int(np.argmax(areas)) if areas else 0
                kp    = kps_all[best].cpu().numpy()

                orientation = detect_orientation(kp)
                result      = analyzer.analyze(kp, orientation=orientation)
                draw_skeleton(frame, kp, result["status"])

                for ang_name, ang_val in result["angles"].items():
                    if ang_name in ANGLE_KP:
                        li, ri = ANGLE_KP[ang_name]
                        pt = (bilateral_kp(kp, li, ri) if orientation == "front"
                              else best_side_kp(kp, li, ri))
                        draw_angle_label(frame, pt, ang_val)

        # Flash verde al completar rep
        if result["reps"] > getattr(analyzer, "_last_rep_drawn", -1):
            setattr(analyzer, "_last_rep_drawn", result["reps"])
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (frame_w, frame_h), C["ok"], -1)
            cv2.addWeighted(flash, 0.13, frame, 0.87, 0, frame)
            cv2.rectangle(frame, (2, 2), (frame_w-2, frame_h-2), C["ok"], 5)

        draw_hud(frame, result, exercise_name, fps)
        out.write(frame)

        # Progreso en consola (cada 30 frames)
        if frame_idx % 30 == 0 or frame_idx == 1:
            elapsed = time.time() - t_start
            pct     = f"{frame_idx/total_f*100:.1f}%" if total_f else "?"
            msgs    = " | ".join(m for m, _ in result["feedback"][:1])
            orient_s = result.get("orientation","side")[0].upper()
            print(f"\r  [{pct:>6}] Frame {frame_idx:5d}  "
                  f"Reps:{result['reps']:3d}  [{orient_s}]  "
                  f"{result['status'].upper():8s}  {msgs}      ",
                  end="", flush=True)

        # Solo mostrar ventana si es webcam
        if use_display:
            cv2.imshow(f"Pose Analyzer - {exercise_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[!] Salida por usuario.")
                break

    cap.release()
    out.release()
    if use_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n\n{'='*55}")
    print(f"  PROCESADO FINALIZADO")
    print(f"  Ejercicio      : {exercise_name}")
    print(f"  Repeticiones   : {analyzer.reps}")
    print(f"  Frames proc.   : {frame_idx}")
    print(f"  Tiempo total   : {elapsed:.1f}s")
    print(f"  Video anotado  : {out_path}")
    print(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════════════
#  ENTRADA
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Analyzer — YOLOv8-Pose + yt_dlp")
    parser.add_argument("--url",      default=DEFAULT_URL)
    parser.add_argument("--source",   default=None,
                        help="Fuente local: 0=webcam, ruta a video. Ignora --url si se usa.")
    parser.add_argument("--exercise", default="squat",
                        choices=["squat", "deadlift", "pushup"])
    parser.add_argument("--model",    default="n", choices=["n", "s", "m", "l"])
    parser.add_argument("--conf",     default=0.50, type=float)

    args   = parser.parse_args()
    source = args.source if args.source is not None else args.url

    run(source=source, exercise=args.exercise,
        model_size=args.model, conf=args.conf)