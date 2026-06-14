# main.py
import os
import re
import math
import shutil
import zipfile
import logging
import warnings
import gc  # لتنظيف الذاكرة فوراً
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Precise ATS AI API Lite", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # قمنا بإلغاء النموذج الثاني الضخم لتوفير الذاكرة ومنع الـ Out of Memory
    WEIGHTS = {
        "semantic": 0.65,  # رفعنا وزن التطابق الدلالي لتعويض النموذج الملغى
        "skill": 0.20,
        "experience": 0.15
    }

    SKILL_ALIASES = {
        "python": ["python", "py"], "tensorflow": ["tensorflow", "tf"], "pytorch": ["pytorch"],
        "javascript": ["javascript", "js"], "react": ["react", "reactjs"], "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning", "dl"], "nlp": ["nlp", "natural language processing"], "sql": ["sql"],
        "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "aws": ["aws"], "linux": ["linux"],
        "git": ["git"], "cybersecurity": ["cybersecurity", "cyber security"], "devops": ["devops"],
        "api": ["api", "rest api"], "fastapi": ["fastapi"], "django": ["django"], "flask": ["flask"],
        "java": ["java"], "csharp": ["c#", ".net", "asp.net"], "html": ["html"], "css": ["css"],
        "nodejs": ["nodejs", "node.js"], "mongodb": ["mongodb"], "flutter": ["flutter"], "firebase": ["firebase"],
        "android": ["android"], "kotlin": ["kotlin"], "terraform": ["terraform"], "networking": ["networking"],
        "wireshark": ["wireshark"], "nmap": ["nmap"], "pentesting": ["pentesting"], "typescript": ["typescript", "ts"],
        "bootstrap": ["bootstrap"], "tailwind css": ["tailwind css", "tailwind"], "sass": ["sass"], "less": ["less"],
        "angular": ["angular", "angularjs"], "vue.js": ["vue.js", "vue", "vuejs"], "next.js": ["next.js", "nextjs"],
        "nuxt.js": ["nuxt.js", "nuxtjs"], "svelte": ["svelte"], "blazor": ["blazor"], "jquery": ["jquery"]
    }

    SKILL_ALIAS_KEY_TO_ID = {
        "python": 177, "javascript": 178, "react": 186, "java": 176, "csharp": 175, "html": 180, "css": 181,
        "nodejs": 227, "mongodb": 280, "kotlin": 201, "django": 237, "flask": 238, "fastapi": 239, "sql": 271,
        "git": 301, "docker": 317, "kubernetes": 318, "aws": 335, "linux": 323, "firebase": 282, "networking": 48,
        "cybersecurity": 31, "devops": 30, "api": 25, "tensorflow": 373, "pytorch": 374, "machine learning": 375,
        "deep learning": 376, "nlp": 377, "flutter": 378, "android": 379, "terraform": 380, "wireshark": 381,
        "nmap": 382, "pentesting": 383, "typescript": 179, "bootstrap": 182, "tailwind css": 183, "sass": 184,
        "less": 185, "angular": 187, "vue.js": 188, "next.js": 189, "nuxt.js": 190, "svelte": 191, "blazor": 192, "jquery": 193
    }

logger.info("Loading lightweight Embedding model...")
embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
logger.info("Model loaded inside memory limits successfully.")

def parse_date_safely(text_context):
    months_map = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
    numeric_range = re.findall(r"\b(0[1-9]|1[0-2])[-/](20\d{2}|19\d{2})\b", text_context)
    if numeric_range: return f"{numeric_range[0][1]}-{numeric_range[0][0]}-01"
    for m_name, m_num in months_map.items():
        if re.search(rf"\b{m_name}\b", text_context, re.IGNORECASE):
            yr = re.search(r"\b(20\d{2}|19\d{2})\b", text_context)
            if yr: return f"{yr.group(1)}-{m_num}-01"
    yr_only = re.search(r"\b(20\d{2}|19\d{2})\b", text_context)
    if yr_only: return f"{yr_only.group(1)}-01-01"
    return None

def clean_extracted_name(name, strip_words):
    if not name: return "Not Found"
    name = re.sub(r"[:\-,\.\(\)\t]", " ", name)
    for word in strip_words: name = re.sub(rf"\b{word}\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name if len(name) > 2 else "Not Found"

class Extractor:
    def __init__(self):
        self.skill_patterns = {s: [re.compile(rf"(?<![a-zA-Z0-9]){re.escape(a)}(?![a-zA-Z0-9])", re.IGNORECASE) for a in als] for s, als in Config.SKILL_ALIASES.items()}
    def extract_skills(self, text):
        f_ids = set()
        for s, pats in self.skill_patterns.items():
            if any(p.search(text) for p in pats):
                sid = Config.SKILL_ALIAS_KEY_TO_ID.get(s.lower())
                if sid is not None: f_ids.add(sid)
        return sorted(list(f_ids))
    def extract_years(self, text):
        lt = text.lower()
        yf = []
        pats = [r"(\d{1,2})\+?\s*(?:years|yrs)\s+(?:of\s+)?experience", r"experience\s*(?:of|in)?\s*(\d{1,2})\+?\s*(?:years|yrs)?"]
        for p in pats:
            for x in re.findall(p, lt):
                v = int(x)
                if 0 <= v <= 40: yf.append(v)
        return max(yf) if yf else 0

def extract_education_info(text):
    edu_list = []
    seen = set()
    deg_levels = [{"level": "PhD", "patterns": [r"\b(ph\.?d\.?|doctorate)\b"]}, {"level": "Master", "patterns": [r"\b(master(?:'s)?)\b"]}, {"level": "Bachelor", "patterns": [r"\b(bachelor(?:'s)?)\b"]}]
    uni_pats = [r"\b(University\s+of\s+[A-Z][A-Za-z&.\s]{2,40})\b", r"\b([A-Z][A-Za-z&.\s]{2,40}\sUniversity)\b"]
    gpa_pats = [r"\b(?:gpa|cgpa|grade)\s*[:\-]?\s*([0-4]\.\d{1,3}|4\.0)\b"]

    lines = text.split('\n')
    for line in lines:
        if any(k in line.lower() for k in ["education", "university", "bachelor", "master", "degree", "college"]):
            inst, deg, gpa, fd, td = None, None, None, None, None
            for pat in uni_pats:
                m = re.search(pat, line)
                if m: inst = m.group(1).strip(); break
            if inst:
                inst = clean_extracted_name(inst, ["From", "To", "Date", "GPA"])
                if inst.lower() in seen or inst == "Not Found": continue
            for g in deg_levels:
                for pat in g["patterns"]:
                    if re.search(pat, line, re.IGNORECASE): deg = g["level"]; break
                if deg: break
            for pat in gpa_pats:
                m = re.search(pat, line, re.IGNORECASE)
                if m: gpa = m.group(1).strip(); break
            yrs = re.findall(r"\b(20\d{2}|19\d{2})\b", line)
            if len(yrs) >= 2: fd, td = f"{yrs[0]}-01-01", f"{yrs[1]}-01-01"
            elif len(yrs) == 1: td = f"{yrs[0]}-01-01"
            if inst and inst != "Not Found":
                seen.add(inst.lower())
                edu_list.append({"institutionName": inst, "fromDate": fd, "toDate": td, "degree": deg if deg else "Bachelor", "gpa": gpa})
    return edu_list

def extract_experience_info(text, ext):
    exp_list = []
    seen = set()
    comp_pats = [r"\b([A-Z][A-Za-z0-9\s&.,]{2,30}\s(?:Ltd|Inc|Corp|Company|Solutions|Bank|Center|Institute|Group))\b"]
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in ["engineer", "developer", "analyst", "manager", "intern"]):
            comp, fd, td, title = "Not Found", None, None, line.strip()[:60]
            ctx = line + " " + (lines[i+1] if i+1 < len(lines) else "")
            for pat in comp_pats:
                m = re.search(pat, ctx)
                if m: comp = m.group(1).strip(); break
            if comp:
                comp = clean_extracted_name(comp, ["Experience", "Using", "Present"])
                if comp.lower() in seen or comp == "Not Found": continue
            dr = re.search(r"\b(20\d{2}|19\d{2})\s*[-–—]\s*(20\d{2}|19\d{2}|present|current|now)\b", ctx, re.IGNORECASE)
            if dr:
                fd = f"{dr.group(1)}-01-01"
                td = "2026-01-01" if any(w in dr.group(2).lower() for w in ["present", "current", "now"]) else f"{dr.group(2)}-01-01"
            else: fd = parse_date_safely(ctx)
            if comp != "Not Found":
                seen.add(comp.lower())
                exp_list.append({"companyName": comp, "fromDate": fd, "toDate": td, "jobTitle": title, "description": ctx.strip()[:150] + "...", "skills": ext.extract_skills(ctx)})
    return exp_list[:5]

@app.post("/api/v1/screen-cvs")
async def screen_cvs_endpoint(roleName: str = Form(...), jobDesc: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"): raise HTTPException(status_code=400, detail="ZIP only.")
    tdir, exdir = "./t_zip", f"./cvs_{int(random.randint(1000, 9999))}"
    os.makedirs(tdir, exist_ok=True); os.makedirs(exdir, exist_ok=True)
    zpath = os.path.join(tdir, file.filename)
    try:
        with open(zpath, "wb") as b: shutil.copyfileobj(file.file, b)
        with zipfile.ZipFile(zpath, 'r') as zr: zip_ref = zr; zr.extractall(exdir)
        ext = Extractor(); api_results = []
        for r, ds, fls in os.walk(exdir):
            for cf in fls:
                cpath = os.path.join(r, cf); txt = ""
                if cf.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(cpath)
                        txt = "\n".join([page.extract_text() or "" for page in reader.pages])
                    except: continue
                elif cf.lower().endswith(".docx"):
                    try:
                        doc = docx.Document(cpath)
                        txt = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    except: continue
                elif cf.lower().endswith(".txt"):
                    try:
                        with open(cpath, "r", encoding="utf-8", errors="ignore") as f: txt = f.read()
                    except: continue
                txt = re.sub(r"\s+", " ", str(txt)).strip()
                if not txt: continue
                
                s_ids = ext.extract_skills(txt)
                yrs = ext.extract_years(txt)
                js_ids = set(ext.extract_skills(jobDesc))
                cv_s_ids = set(s_ids)
                m_ids = sorted(list(cv_s_ids & js_ids))
                ms_ids = sorted(list(js_ids - cv_s_ids))
                
                c_emb = embedder.encode([txt.lower()])
                j_emb = embedder.encode([jobDesc.lower()])
                sem_score = max(0.0, float(cosine_similarity(c_emb, j_emb)[0][0]))
                
                s_score = len(m_ids) / len(js_ids) if js_ids else 0.0
                e_score = min(yrs / 10.0, 1.0)
                f_score = (sem_score * Config.WEIGHTS["semantic"] + s_score * Config.WEIGHTS["skill"] + e_score * Config.WEIGHTS["experience"])
                
                api_results.append({
                    "applicationId": int(hash(cf) % 1000000), "finalScore": round(f_score, 4), "mlScore": round(f_score, 4), "semanticScore": round(sem_score, 4),
                    "aiExplanation": f"Matches {len(m_ids)} skills. Semantic: {sem_score*100:.1f}%. Exp: {yrs} yrs.", "linkedInUrl": None,
                    "position": roleName, "resubmissionRequired": bool(f_score < 0.45), "matchingSkills": [int(x) for x in m_ids], "missingSkills": [int(x) for x in ms_ids],
                    "experienceSummary": extract_experience_info(txt, ext), "educationSummary": extract_education_info(txt)
                })
        api_results = sorted(api_results, key=lambda x: x["finalScore"], reverse=True)
        return api_results
    finally:
        if os.path.exists(zpath): os.remove(zpath)
        if os.path.exists(exdir): shutil.rmtree(exdir)
        gc.collect()  # تفريغ الذاكرة فوراً لمنع الـ Out of Memory
