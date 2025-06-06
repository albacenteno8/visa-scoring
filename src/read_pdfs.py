import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm
import re
import unicodedata

# initialize Paddle
ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log=False)

def extract_banners_from_pdf(pdf_path):
    super_headers = []

    doc = fitz.open(pdf_path)
    for page in doc:
        # Render page -> image
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if w < 200 or h < 20:
                continue
            crop = img[y:y+h, x:x+w]
            black_ratio = cv2.countNonZero(thresh[y:y+h, x:x+w]) / float(w*h)
            if black_ratio < 0.6:
                continue

            inv = cv2.bitwise_not(crop)
            try:
                result = ocr.ocr(inv, cls=True)
            except Exception:
                continue
            if not result:
                continue

            # Safely extract all recognized text lines
            lines = []
            for block in result:
                if not block:
                    continue
                for line in block:
                    # line is typically [ [x1,y1], [text, conf], ... ]
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        lines.append(line[1][0])
            text = " ".join(lines).strip()
            if not text:
                continue

            # classify banner size
            if h > 50:
                super_headers.append(text)

    # dedupe while preserving order
    uniq_super = list(dict.fromkeys(super_headers))

    return uniq_super

def scan_folder_and_save_csv(root_folder : str, out_headers : str, check_folders : int = None):
    all_super = []
    
    if check_folders is not None:
        list_items = list(set(os.listdir(root_folder)) - set(['.DS_Store']))[0:check_folders]
    else:
        list_items = list(set(os.listdir(root_folder)) - set(['.DS_Store']))

    for case in tqdm(sorted(list_items), desc="Folders"):
        case_path = os.path.join(root_folder, case)
        if not os.path.isdir(case_path):
            continue
        # find the application PDF
        excludes = {'acknowledgement', 'withdrawal', 'bvb', 'rdv', 'passport','summary'}
        pdfs = [f for f in os.listdir(case_path) if (name := f.lower()).endswith(".pdf")
                and "application" in name and not any(ex in name for ex in excludes)]
        if not pdfs:
            continue
        pdf_path = os.path.join(case_path, pdfs[0])
        sh = extract_banners_from_pdf(pdf_path)
        all_super.extend(sh)
    
    # dedupe
    uniq_super = sorted(set(all_super))
    
    # save
    pd.Series(uniq_super, name="super_header").to_csv(out_headers, index=False)
    
    print(f"Saved {len(uniq_super)} super‐headers to {out_headers}")
    return uniq_super

def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")

def initials(text: str) -> str:
    return "".join(w[0] for w in text.split() if w).lower()

def process_applications(root_folder: str, subheaders_list: list[str], check_folders : int = None):
    sub_set = {s.lower() for s in subheaders_list}

    rows = []
    if check_folders is not None: 
        list_items = list(set(os.listdir(root_folder)) - set(['.DS_Store']))[0:check_folders]
    else:
        list_items = list(set(os.listdir(root_folder)) - set(['.DS_Store']))

    for case in tqdm(sorted(list_items), desc="Cases"):
        case_path = os.path.join(root_folder, case)
        if not os.path.isdir(case_path):
            continue

        # find application PDF
        app_pdf = None
        excludes = {'acknowledgement', 'withdrawal', 'bvb', 'passport', 'summary'}
        for fn in os.listdir(case_path):
            low = fn.lower()
            if ( "application" in low and low.endswith(".pdf")  and not any(ex in low for ex in excludes)):
                app_pdf = os.path.join(case_path, fn)
                break
        if not app_pdf:
            continue

        # collect all spans
        doc = fitz.open(app_pdf)
        spans = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        font = span.get("font", span.get("fontname", "")).lower()
                        spans.append({
                            "text": span["text"].strip(),
                            "size": round(span["size"], 1),
                            "flags": span["flags"],
                            "font": font,
                        })

        # counters
        ofm_count = 0
        cv_count = 0
        ehd_count = 0

        # count occurrences of Employment history details anywhere
        for s in spans:
            if s["text"].strip().lower() == "employment history details":
                ehd_count += 1

        current_sup = None
        current_sub = None
        rec = {"sample_id": case}

        for i, s in enumerate(spans):
            t = s["text"]
            if not t:
                continue

            bold = bool(s["flags"] & 2) or ("bold" in s["font"])
            low = t.lower()

            # super-header?  size=14.0 flags=16 bold font
            if bold and s["size"] == 14.0 and s["flags"] == 16:
                current_sup = t
                current_sub = None
                if low == "other family members":
                    ofm_count += 1
                if low == "country visited":
                    cv_count += 1
                continue

            # sub-header?
            if bold and low in sub_set:
                current_sub = t
                continue

            # field/question?
            is_field = (not bold) and (t.endswith("?") or ":" in t)
            if is_field:
                # skip if the question looks like page footer or layout artifact
                if any(kw in t.lower() for kw in ("generated", "reference number", "trn")):
                    continue
                
                # sanitize parentheses
                key_txt = re.sub(r"\(.*?\)", "X", t)
                base = key_txt.split(":", 1)[0].rstrip("?").strip()
                col_slug = slugify(base)
                parts = [col_slug]
                if current_sup:
                    parts.append(initials(current_sup))
                if current_sub:
                    parts.append(initials(current_sub))
                colname = "_".join(parts)

                # find next bold span as valid answer
                ans = ""
                for nxt in spans[i + 1:]:
                    nxt_bold = bool(nxt["flags"] & 2) or ("bold" in nxt["font"])
                    val = nxt["text"].strip()
                    if nxt_bold and val and val.lower() not in ("personal privacy", "official: sensitive"):
                        ans = val
                        break
                rec[colname] = ans
                continue

            # special: Value in Australian dollars
            if "value in australian dollars" in low:
                val = ""
                for nxt in spans[i + 1:]:
                    nxt_bold = bool(nxt["flags"] & 2) or ("bold" in nxt["font"])
                    if nxt_bold and re.match(r"^[\d,\.]+$", nxt["text"].replace(" ", "")):
                        val = nxt["text"].strip()
                        break
                if val:
                    rec["value_in_australian_dollars"] = val
                continue
        
        # add generation date from first page
        generated_value = None
        for span in spans[:100]:  # assuming ~100 spans on first page is enough
            txt = span["text"]
            if txt.lower().startswith("generated:"):
                generated_value = txt.split(":", 1)[-1].strip()
                break
        rec["generated"] = generated_value
        
        # store counts
        rec["other_family_members_count"] = ofm_count
        rec["country_visited_count"] = cv_count
        rec["employment_history_details_count"] = ehd_count

        rows.append(rec)

    df = pd.DataFrame(rows).set_index("sample_id")
    return df