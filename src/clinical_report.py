"""
Klinik PDF raporu — model çıktısından (run_inference sonucu) hastaya özel üretilir.

İçerik TÜBİTAK Raporu Md. 2.5 Click-to-Explain kartıyla aynı bilgileri taşır: etkilenen duvar(lar),
ST sapması (mm), şüpheli koroner arter, eksik kanallara rağmen karar güveni. Ayrıca kopan kanalların
hangi sağlam kanallardan yeniden üretildiği (Spatial Attention) ve Einthoven ile tamamlanan kanallar yazılır.

ReportLab'in standart Helvetica yazı tipi ş, ğ, ı, İ karakterlerini içermediğinden metinler ASCII'ye çevrilir.
"""
import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from src.ecg_constants import WALL_GROUPS, WALL_NAMES

WALL_TR = {"Anterior": "On Duvar (Anterior)", "Inferior": "Alt Duvar (Inferior)", "Lateral": "Yan Duvar (Lateral)"}
WALL_TR_SHORT = {"Anterior": "on", "Inferior": "alt", "Lateral": "yan"}

_ASCII = str.maketrans("şŞğĞıİçÇöÖüÜ∣–—", "sSgGiIcCoOuU|--")

NAVY = (0.08, 0.22, 0.38)
TEXT = (0.15, 0.15, 0.15)
MUTED = (0.4, 0.4, 0.45)
RED = (0.75, 0.1, 0.1)
GREEN = (0.1, 0.45, 0.2)

LOW_CONFIDENCE = 0.70  # bu güvenin altındaki kararlar "sınırda" olarak işaretlenir

PAGE_W, PAGE_H = letter
LEFT, RIGHT = 45, PAGE_W - 45
WIDTH = RIGHT - LEFT


def tr(text):
    return str(text).translate(_ASCII)


def _pct(x):
    return f"%{100 * x:.0f}"


def _fmt_mm(v):
    return "-" if v is None else f"{v:+.1f}"


class _Writer:
    """Sayfa taşmasını yöneten basit satır yazıcı."""

    def __init__(self, c):
        self.c = c
        self.y = PAGE_H - 110

    def ensure(self, h):
        if self.y - h < 50:
            self.c.showPage()
            self.y = PAGE_H - 50

    def heading(self, text, color=NAVY):
        self.ensure(40)
        self.y -= 14
        self.c.setFillColorRGB(*color)
        self.c.setFont("Helvetica-Bold", 12)
        self.c.drawString(LEFT, self.y, tr(text))
        self.y -= 7
        self.c.setStrokeColorRGB(0.8, 0.8, 0.8)
        self.c.line(LEFT, self.y, RIGHT, self.y)
        self.y -= 14

    def para(self, text, font="Helvetica", size=9.5, color=TEXT, indent=10, gap=3):
        lines = simpleSplit(tr(text), font, size, WIDTH - indent)
        self.ensure(len(lines) * (size + 3) + gap)
        self.c.setFillColorRGB(*color)
        self.c.setFont(font, size)
        for line in lines:
            self.c.drawString(LEFT + indent, self.y, line)
            self.y -= size + 3
        self.y -= gap

    def kv(self, key, value, color=TEXT):
        self.para(f"{key}: {value}", color=color)


def _header(c, result):
    c.setFillColorRGB(*NAVY)
    c.rect(0, PAGE_H - 80, PAGE_W, 80, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(LEFT, PAGE_H - 35, "12-Lead ECG MAE Clinical Decision & XAI Report")
    c.setFont("Helvetica", 9)
    c.drawString(LEFT, PAGE_H - 55, tr("Yapay Zeka Destekli Kardiyak Karar Destek ve Eksik Kanal Tolerans Sistemi"))
    record = f"Kayit: PTB-XL #{result['ecg_id']}" if result.get("ecg_id") else "Kayit: Kullanici yuklemesi"
    c.drawRightString(RIGHT, PAGE_H - 55, record)
    c.drawRightString(RIGHT, PAGE_H - 68, datetime.now().strftime("%d.%m.%Y %H:%M"))


def _wall_table(w, result):
    cards = result["click_to_explain"]
    cols = [("Duvar", 0), ("Derivasyonlar", 105), ("Hasar", 205), ("Maks. ST (mm)", 250),
            ("Supheli arter", 325), ("Kopuk kanal", 455)]
    w.ensure(18 * (len(WALL_NAMES) + 1) + 10)
    c = w.c
    c.setFillColorRGB(0.93, 0.94, 0.96)
    c.rect(LEFT, w.y - 4, WIDTH, 16, fill=True, stroke=False)
    c.setFillColorRGB(*NAVY)
    c.setFont("Helvetica-Bold", 8.5)
    for name, x in cols:
        c.drawString(LEFT + 5 + x, w.y + 1, name)
    w.y -= 18
    for wall in WALL_NAMES:
        card = cards[wall]
        color = RED if card["affected"] else TEXT
        c.setFillColorRGB(*color)
        c.setFont("Helvetica-Bold" if card["affected"] else "Helvetica", 8.5)
        artery = tr(card["suspected_artery"])
        if len(artery) > 30:
            artery = artery[:29] + "."
        values = [WALL_TR[wall], ", ".join(WALL_GROUPS[wall]), _pct(card["damage_score"]),
                  _fmt_mm(card["max_st_deviation_mm"]), artery,
                  ", ".join(card["missing_leads"]) or "-"]
        for (_, x), v in zip(cols, values):
            c.drawString(LEFT + 5 + x, w.y, tr(v))
        w.y -= 16
    w.y -= 6


def build_clinical_pdf(result):
    """result: main.run_inference çıktısı (+ ecg_id, ground_truth). Döndürür: PDF içeren BytesIO."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    _header(c, result)
    w = _Writer(c)

    diag = result["diagnosis"]
    dropped = result["dropped_lead_names"]
    einthoven = result.get("einthoven_completed_leads", [])
    sources = result.get("reconstruction_sources", {})
    walls = diag.get("affected_walls") or []

    # 1. Sistem güvenilirliği ve eksik kanal durumu
    w.heading("1. Sistem Guvenilirligi ve Eksik Kanal Durumu")
    w.kv("Aktif kanal", f"{result['active_lead_count']} / 12 derivasyon")
    w.kv("Kopuk kanallar", ", ".join(dropped) if dropped else "Yok (tam 12 kanal)")
    for lead in dropped:
        if lead in einthoven:
            w.para(f"- {lead}: saglam ekstremite derivasyonlarindan Einthoven bagintilariyla kesin hesaplandi.", indent=20)
        elif lead in sources:
            src = ", ".join(f"{k} {_pct(v)}" for k, v in sources[lead].items())
            w.para(f"- {lead}: MAE ile yeniden uretildi; en cok temsil eden saglam kanallar: {src}.", indent=20)
    w.kv("Karar esigi / cikarim suresi", f"{result['xai_metrics']['threshold']:.2f} / {result['inference_ms']} ms")

    # 2. Klinik teşhis
    w.heading("2. Klinik Teshis")
    w.kv("Tani tahmini", diag["predicted_class"], color=RED if diag["is_mi"] else GREEN)
    w.kv("MI olasiligi / karar guveni", f"{_pct(result['mi_probability'])} / {_pct(diag['confidence'])}")
    if walls:
        w.kv("Etkilenen duvar(lar)", ", ".join(WALL_TR[x] for x in walls))
    w.y -= 4
    _wall_table(w, result)

    # 3. Hekim için XAI yorumu
    w.heading("3. Hekim Icin XAI (Aciklanabilir Yapay Zeka) Yorumu")
    if diag["is_mi"]:
        for wall in walls:
            w.para(result["click_to_explain"][wall]["explanation"], font="Helvetica-Oblique")
    else:
        w.para(f"Model MI lehine bulgu saptamadi (MI olasiligi {_pct(result['mi_probability'])}, "
               f"karar esigi {result['xai_metrics']['threshold']:.2f}).", font="Helvetica-Oblique")
    if diag["confidence"] < LOW_CONFIDENCE:
        w.para(f"DIKKAT: Karar sinirda (guven {_pct(diag['confidence'])}, MI olasiligi {_pct(result['mi_probability'])}, "
               f"esik {result['xai_metrics']['threshold']:.2f}). Sonuc hekim tarafindan EKG uzerinde dogrudan "
               "degerlendirilmelidir.", font="Helvetica-Bold", color=RED)
    xai = result["xai_metrics"]
    w.para(f"Spatial Attention katmaninda en cok dikkat alan derivasyon: {xai['most_affected_lead']} "
           f"({xai['anatomical_region']}, agirlik {_pct(xai['max_lead_attention'])}).", font="Helvetica-Oblique")
    if dropped:
        w.para(f"{len(dropped)} kanal kopuk olmasina ragmen karar guveni {_pct(diag['confidence'])}. "
               "Kopuk kanallar saglam kanallardan yeniden uretilmistir.", font="Helvetica-Oblique")
    w.para("ST sapmasi, J noktasindan yaklasik 60 ms sonra PR segmentine gore olculur (1 mm = 0.1 mV); "
           "aciklama amaclidir.", size=8, color=MUTED)

    # 4. Hasta bilgilendirme notu
    w.heading("4. Hasta ve Hasta Yakini Bilgilendirme Notu", color=GREEN)
    w.para("Bu rapor, EKG sinyallerinizin yapay zeka tarafindan incelenmesiyle olusturulmustur.")
    if diag["is_mi"]:
        names = " ve ".join(WALL_TR_SHORT[x] for x in walls)
        w.para(f"Inceleme sonucunda kalbinizin {names} duvar{'larina' if len(walls) > 1 else 'ina'} ait elektriksel "
               "sinyallerde hekim kontrolu gerektiren bulgular izlenmistir. Lutfen en kisa surede bir "
               "kardiyoloji uzmanina basvurunuz.", font="Helvetica-Bold", color=RED)
    else:
        w.para("EKG'nizde kalp krizi (miyokard enfarktusu) lehine belirgin bir bulgu izlenmemistir.")
    if diag["confidence"] < LOW_CONFIDENCE:
        w.para("Yapay zekanin bu kayittaki karari kesin degildir; hekim degerlendirmesi ozellikle onemlidir.")
    if dropped:
        w.para(f"Cekim sirasinda {len(dropped)} elektrottan sinyal alinamamis; sistem eksik sinyalleri diger "
               "elektrotlardan tamamlamistir.")
    w.para("Lutfen bu on degerlendirme sonucunu kardiyoloji uzman hekiminizle paylasiniz.")

    # Alt bilgi
    w.y -= 6
    w.para("Bu rapor yapay zeka destekli bir on degerlendirmedir; kesin tani hekim tarafindan konur.",
           size=8, color=MUTED, indent=0)
    gt = result.get("ground_truth")
    if gt:
        walls_gt = ", ".join(gt["mi_walls"]) or "-"
        w.para(f"Demo kaydi - PTB-XL gercek etiketi: {gt['label']} (duvar: {walls_gt})", size=8, color=MUTED, indent=0)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
