"""
Rapor metinleri — model çıktısından (run_inference sonucu) hastaya özel üretilir (Rapor Md. 2.5 Click-to-Explain).

  build_narrative(result) -> {
      "summary":          tek cümlelik hekim özeti,
      "clinician":        hekim için madde madde değerlendirme (duvar, ST, arter, dikkat, kopuk kanallar),
      "patient":          hasta ve yakını için sade dilde açıklama,
      "warning":          düşük güven uyarısı (yoksa None),
      "confidenceNote":   "model güveni" ile "duvar olasılığı" arasındaki farkın açıklaması,
  }
Sabit şablon metin yoktur: her cümle o kaydın teşhisine, ST ölçümüne, dikkat ağırlıklarına ve kopuk kanallarına göre kurulur.
"""
from src.ecg_constants import WALL_GROUPS, WALL_NAMES

LOW_CONFIDENCE = 0.70
WALL_CLINICAL = {"Anterior": "Ön duvar (anterior)", "Inferior": "Alt duvar (inferior)", "Lateral": "Yan duvar (lateral)"}
WALL_PLAIN = {"Anterior": "ön", "Inferior": "alt", "Lateral": "yan"}


def _pct(x):
    return f"%{100 * x:.0f}"


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


def _peak_st(card):
    """Duvarın derivasyonları içinde mutlak değerce en büyük ST sapması: (derivasyon, mm) ya da None."""
    vals = [(lead, v) for lead, v in card["st_deviation_mm"].items() if v is not None]
    return max(vals, key=lambda t: abs(t[1])) if vals else None


def build_narrative(result):
    diag = result["diagnosis"]
    cards = result["click_to_explain"]
    walls = diag.get("affected_walls") or []
    dropped = result["dropped_lead_names"]
    einthoven = set(result.get("einthoven_completed_leads", []))
    sources = result.get("reconstruction_sources", {})
    mi_p, conf, thr = result["mi_probability"], diag["confidence"], result["xai_metrics"]["threshold"]
    low = conf < LOW_CONFIDENCE

    # --- Hekim -------------------------------------------------------------------------------------
    clinician = []
    if diag["is_mi"]:
        summary = (f"{_join(WALL_CLINICAL[w].lower() for w in walls).capitalize()} tutulumuyla uyumlu MI şüphesi "
                   f"(MI olasılığı {_pct(mi_p)}).")
        for w in walls:
            card, peak = cards[w], _peak_st(cards[w])
            st = f"en belirgin ST sapması {peak[1]:+.1f} mm ({peak[0]})" if peak else "ST ölçülemedi"
            line = f"{WALL_CLINICAL[w]}: hasar skoru {_pct(card['damage_score'])}, {st}; şüpheli arter {card['suspected_artery']}."
            if card["missing_leads"]:
                line += (f" Bu duvarın {len(card['missing_leads'])}/{len(WALL_GROUPS[w])} derivasyonu kopuk "
                         f"({', '.join(card['missing_leads'])}); bu duvar için karar daha az kesindir.")
            clinician.append(line)
        others = [w for w in WALL_NAMES if w not in walls and cards[w]["damage_score"] >= 0.25]
        for w in others:
            clinician.append(f"{WALL_CLINICAL[w]} için de sınırda bulgu var (hasar skoru {_pct(cards[w]['damage_score'])}); "
                             "eşik altında kaldığı için tanıya eklenmedi.")
    else:
        summary = f"MI lehine bulgu saptanmadı (MI olasılığı {_pct(mi_p)}, karar eşiği {_pct(thr)})."

    xai = result["xai_metrics"]
    clinician.append(f"Model kararında en çok {xai['most_affected_lead']} derivasyonuna dayandı "
                     f"({xai['anatomical_region']}; Spatial Attention ağırlığı {_pct(xai['max_lead_attention'])}).")

    if dropped:
        parts = []
        for lead in dropped:
            if lead in einthoven:
                parts.append(f"{lead} → sağlam ekstremite derivasyonlarından Einthoven bağıntısıyla kesin hesaplandı")
            elif lead in sources:
                parts.append(f"{lead} → MAE ile " + ", ".join(f"{k} ({_pct(v)})" for k, v in sources[lead].items())
                             + " kanallarından üretildi")
        clinician.append(f"{len(dropped)} derivasyon kopuk: " + "; ".join(parts) + ".")
        clinician.append(f"Kopuk kanallara rağmen karar güveni {_pct(conf)}.")

    warning = None
    if low:
        warning = (f"Karar sınırda: güven {_pct(conf)}, MI olasılığı {_pct(mi_p)} (karar eşiği {_pct(thr)}). "
                   "Sonuç hekim tarafından EKG üzerinde doğrudan değerlendirilmelidir.")

    # --- Hasta -------------------------------------------------------------------------------------
    patient = ["Bu rapor, EKG kaydınızın yapay zekâ tarafından incelenmesiyle hazırlanmış bir ön değerlendirmedir."]
    if diag["is_mi"]:
        names = _join(WALL_PLAIN[w] for w in walls)
        patient.append(f"Kalbinizin {names} duvar{'larına' if len(walls) > 1 else 'ına'} ait elektriksel sinyallerde, "
                       "kalp krizi (miyokard enfarktüsü) ile uyumlu olabilecek değişiklikler görüldü.")
        patient.append("Bu bulgu acil değerlendirme gerektirebilir. Lütfen vakit kaybetmeden bir hekime başvurunuz.")
    else:
        patient.append("EKG'nizde kalp krizi (miyokard enfarktüsü) ile uyumlu belirgin bir bulgu görülmedi.")
        patient.append("Göğüs ağrısı, nefes darlığı veya çarpıntı gibi şikâyetleriniz devam ederse yine de hekime başvurunuz.")
    if dropped:
        patient.append(f"Çekim sırasında {len(dropped)} elektrottan sinyal alınamadı; sistem eksik sinyalleri "
                       "diğer elektrotlardaki bilgiyle tamamladı.")
    if low:
        patient.append("Bu kayıtta yapay zekânın kararı kesin değildir; sonucun hekiminiz tarafından "
                       "değerlendirilmesi özellikle önemlidir.")
    patient.append("Kesin tanıyı yalnızca hekiminiz koyabilir. Bu raporu kardiyoloji uzmanınızla paylaşınız.")

    return {
        "summary": summary,
        "clinician": clinician,
        "patient": patient,
        "warning": warning,
        "confidenceNote": ("Model güveni, MI var/yok kararının olasılığıdır. Duvar olasılıkları ise MI olasılığı ile "
                           "o duvarın tutulma olasılığının çarpımıdır; birden fazla duvar tutulabildiği için "
                           "toplamları %100 olmak zorunda değildir."),
    }

