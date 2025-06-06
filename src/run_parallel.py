import pandas as pd
import numpy as np
import pdfplumber
import re
import unicodedata
import os
from tqdm import tqdm

# Slugs que siempre queremos ignorar
EXCLUDE_SLUGS = [
    'informacion_personal',
    'documentacion_personal',
    'situacion_personal_actual',
    'composicion_familiar',
    'datos_de_contacto',
    'lugar_de_nacimiento',
    'conceptorespuesta',
    'documento_nacional_de_identidad_cedula_de_identidad',
    'fondos_economicos',
    'educacion',
    'datos_del_empleador',
    'trabajos_empleos',
    'historial_migratorio',
    'declaraciones_medicas',
    'antecedentes_penales',
    'documentos_solicitados'
]

INSTRUCTION_PATTERNS = [
    r'^indicar',        # empieza con 'Indica'
    r'^revisa',         # empieza con 'Revisa'
    r'^este documento', # empieza con 'Este documento'
    r'por favor',       # contiene con 'Por favor'
    r'^ejemplo',        # empieza con 'EJEMPLO'
    r'^ej',             # empieza con 'Ej:'
    r'^ten',            # empieza con 'Ten'
    r'^\*',             # empieza con '*',
    r'emitió',          # contiene 'emitió' ,
    r'^solamente',      # empieza con 'Solamente' ,
    r'^puedes',	        # empieza con 'Puedes' ,
    r'^incluye'	        # empieza con 'Incluye' ,
    r'^experiencia',	# empieza con 'Experiencia',
    r'^importante'	    # empieza con 'Importante' ,
]

ATTACHMENT_EXTENSIONS = re.compile(r'\.(pdf|jpe?g|png|rtf|docx)\b', re.IGNORECASE)

def slugify(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    return re.sub(r'[^a-z0-9]+', '_', text).strip('_')


def is_instruction(text: str) -> bool:
    low = text.lower().strip()
    if low.endswith(':'):
        return True
    return any(re.search(pat, low) for pat in INSTRUCTION_PATTERNS)

def sufijos_minuscula(texto):
    # Separa en "palabras" (sean letras mayúsculas) ignorando espacios y caracteres como "/"
    palabras = re.findall(r'[A-ZÑÁÉÍÓÚ]+', texto)
    # Toma la primera letra de cada palabra, pásala a minúscula y únelas
    return ''.join(p[0].lower() for p in palabras)

def clean_question_for_slug(txt: str) -> str:
    """
    Si detecta una pregunta larga con '¿...?' extrae sólo lo que hay entre la última
    abertura '¿' y el signo '?', o bien la última oración que termine en '?'
    """
    q = txt.strip()
    if '¿' in q and '?' in q:
        start = q.rfind('¿')
        end = q.rfind('?')
        q = q[start+1:end]  # quita '¿' y '?'
    elif '?' in q:
        # toma hasta el signo '?'
        part = q[:q.rfind('?')+1]
        # si hay varios puntos, coge lo que viene después del último punto
        if '.' in part:
            q = part.split('.')[-1]
        else:
            q = part
        q = q.strip(' ?')
    # devolver sin signos innecesarios
    return q

def extract_bold_fields_with_labels(pdf_path: str):
    data, labels = {}, {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:
            # Agrupar caracteres por línea
            lines = {}
            for ch in page.chars:
                y = round(ch['top'], 1)
                lines.setdefault(y, []).append(ch)
            parsed = []
            for y in sorted(lines):
                chars = lines[y]
                txt = ''.join(c['text'] for c in sorted(chars, key=lambda c: c['x0'])).strip()
                fonts = [c['fontname'] for c in chars]
                if any('Bold' in f for f in fonts):
                    style = 'bold'
                elif any('Light' in f for f in fonts):
                    style = 'light'
                else:
                    style = 'normal'
                parsed.append((txt, style))

            suffix = None
            for i, (txt, style) in enumerate(parsed):
                # Detectar encabezado de sección
                if txt.isupper() and style == 'bold':
                    suffix = sufijos_minuscula(txt)

                # Extraer campo en negrita válido
                if style == 'bold' and not is_instruction(txt):
                    # limpiamos la pregunta para slug
                    clean_q = clean_question_for_slug(txt)
                    base = slugify(clean_q)
                    if base in EXCLUDE_SLUGS:
                        continue

                    # Construir slug con sufijo de sección si existe
                    slug = f"{base}_{suffix}" if suffix else base
                    labels[slug] = txt

                    # Acumular todas las líneas hasta el siguiente bold
                    values = []
                    for j in range(i+1, len(parsed)):
                        vtxt, vst = parsed[j]
                        # si encontramos otro bold, interrumpimos
                        if vst == 'bold':
                            break
                        # ignorar light e instrucciones
                        if vst == 'light' or is_instruction(vtxt):
                            continue
                        values.append(vtxt)
                    # unir con '; '
                    data[slug] = '; '.join(values).strip()

    return data, labels

def map_attachment_cell(v):
    lv = str(v).lower().strip()
    if re.search(ATTACHMENT_EXTENSIONS, lv):
        return 'yes'
    if 'no se ha subido ningún archivo' in lv:
        return 'no'
    return lv or pd.NA

def extract_text_lines(pdf_path: str):
    """Read all text lines from PDF once."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:
            text = page.extract_text() or ''
            lines.extend([l.strip() for l in text.split('\n') if l.strip()])
    return lines

def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-procesa las celdas que contienen ';' de la siguiente manera:
      - Separa la cadena por ';'
      - Si el último segmento es un entero <100 y el penúltimo es 'si', 'no' o 'yes',
        devuelve el penúltimo.
      - En cualquier otro caso (incluyendo cuando no se cumplan esas condiciones),
        devuelve el valor original.
    """
    def _clean_cell(v):
        if isinstance(v, str) and ';' in v:
            parts = [p.strip() for p in v.split(';') if p.strip()]
            if len(parts) >= 2:
                last = parts[-1]
                try:
                    num = int(last)
                    penult = parts[-2].lower()
                    if num < 100 and penult in ('si', 'no', 'yes'):
                        return parts[-2]  # devolvemos penúltimo con su mayúsc/minúsc original
                    else:
                        if last in ('si', 'no', 'yes'):
                            return last
                        else:
                            return v
                except ValueError:
                    # si el último no es número, devolvemos el valor original completo
                    return v
        return v

    return df.applymap(_clean_cell)

def process_parallel_bold(root_folder: str, field_order=None, field_labels=None):
    """
    Process 'parallel' PDFs in root_folder, returning:
      - DataFrame with one row per PDF
      - Updated field_labels dict mapping slug -> original question text
    """
    if field_order is None:
        field_order = []
    if field_labels is None:
        field_labels = {}

    records = []
    entries = os.listdir(root_folder)
    sample_dirs = [d for d in entries if os.path.isdir(os.path.join(root_folder, d))]
    items = sorted(sample_dirs, key=lambda x: int(x) if x.isdigit() else x) if sample_dirs else sorted(entries)

    for item in tqdm(items, desc="Processing"):
        # Locate PDF
        if os.path.isdir(os.path.join(root_folder, item)):
            folder = os.path.join(root_folder, item)
            pdfs = [f for f in os.listdir(folder) if 'parallel' in f.lower() and f.endswith('.pdf')]
            if not pdfs:
                continue
            pdf_path = os.path.join(folder, pdfs[0])
            sample_id = item
        else:
            if not (item.lower().endswith('.pdf') and 'parallel' in item.lower()):
                continue
            pdf_path = os.path.join(root_folder, item)
            sample_id = os.path.splitext(item)[0]

        # Extract fields and labels
        data, labels = extract_bold_fields_with_labels(pdf_path)

        # Read all lines once for fallback and bullet detection
        text_lines = extract_text_lines(pdf_path)

        # Extraer MONTO:
        monto_val = None
        for line in text_lines:
            m = re.search(r'\bMONTO:\s*(.+)', line, re.IGNORECASE)
            if m:
                monto_val = m.group(1).strip()
                break
        if monto_val is not None:
            data['monto'] = monto_val
            # Asegurarnos de incluirlo en el orden/etiquetas
            if 'monto' not in field_order:
                field_order.append('monto')
                field_labels['monto'] = 'MONTO:'

        # Update field_order and labels with new fields
        for slug, original in labels.items():
            if slug not in field_order:
                field_order.append(slug)
                field_labels[slug] = original

        # Fallback: search only missing slugs in text_lines
        for slug in field_order:
            if slug not in data and slug in field_labels:
                label_text = field_labels[slug].lower()
                for idx, line in enumerate(text_lines):
                    if label_text in line.lower():
                        # Take next non-instruction line as value
                        if idx + 1 < len(text_lines) and not is_instruction(text_lines[idx+1]):
                            data[slug] = text_lines[idx+1]
                        break

        # Exclude unwanted slugs
        for slug in EXCLUDE_SLUGS:
            if slug in field_order:
                field_order.remove(slug)
            field_labels.pop(slug, None)

        # Build row in fixed column order
        row = {slug: data.get(slug) for slug in field_order}
        row['sample_id'] = sample_id
        records.append(row)

    df = pd.DataFrame(records, columns=field_order + ['sample_id'])

    # Poner yes or no en ficheros adjuntos
    attachment_cols = [
        col for col, lbl in field_labels.items()
        if re.search(r'adjuntar|copia|adjunto', lbl, re.IGNORECASE)]
    for col in attachment_cols:
        if col in df.columns:
            df[col] = df[col].apply(map_attachment_cell)
        

    # NaN en lugar de cadena vacía:
    df.replace({'': np.nan}, inplace=True)
    df.replace({r'^(?i:none)$': np.nan}, inplace=True)
    df.replace({'nan': np.nan}, inplace=True)
    df.replace({'none': np.nan}, inplace=True)

    # drops any column where *all* values are NaN
    df = df.dropna(axis=1, how='all')

    return df, field_labels
