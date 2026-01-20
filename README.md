# Proiect Limbaje Formale - Text Summarization cu BART

**Universitatea Tehnică din Cluj-Napoca**  
**Abstractive Summarization folosind facebook/bart-large-cnn**

---

## Descriere

Acest proiect demonstrează **Abstractive Text Summarization** folosind modelul BART (Bidirectional and Auto-Regressive Transformers) de la Facebook/Meta, pre-antrenat pe dataset-ul CNN/DailyMail.

Spre deosebire de **extractive summarization** (care extrage propoziții existente), **abstractive summarization** generează text nou, parafrazează și condensează informația ca un om.

---

## Funcționalități

Interfață web interactivă (Gradio) - foarte ușor de folosit!  
Summarization pe texte în limba engleză  
Control asupra lungimii rezumatului (scurt/mediu/lung)  
3 texte de exemplu pre-încărcate (click și se încarcă)  
Statistici de compresie în timp real  
Input text custom - introdu propriul tău text  
Script pentru terminal (varianta alternativă)  
Comentarii detaliate în română în cod  

---

## Tehnologii folosite

- **Python 3.8+**
- **HuggingFace Transformers** - biblioteca pentru modele NLP
- **PyTorch** - framework de deep learning
- **BART** (`facebook/bart-large-cnn`) - modelul de summarization

---

## Instalare

### 1. Clonează/Descarcă proiectul

```bash
cd c:\Users\Abraham\Desktop\proiect_limbaje
```

### 2. Configurează mediul virtual (RECOMANDAT)

**Opțiunea A - Automată (Windows):**
```cmd
setup_venv.bat
```

**Opțiunea B - Manuală:**
```bash
# Creează mediul virtual
python -m venv venv

# Activează-l
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Instalează dependințele
pip install -r requirements.txt
```

---

## Utilizare

### **Opțiunea A - Interfață Web (RECOMANDAT)**

**Pornire rapidă:**
```cmd
run_app.bat
```

**Sau manual:**
```bash
venv\Scripts\activate
python app.py
```

Aplicația va porni în browser la: **http://127.0.0.1:7860**

---

### **Opțiunea B - Script în Terminal**

```bash
venv\Scripts\activate
python summarization_bart.py
```

---

## Structura proiectului

```
proiapp.py                     # Aplicație web cu interfață Gradio (RECOMANDAT)
├── summarization_bart.py      # Script pentru terminal (alternativă)
├── requirements.txt            # Dependințe Python
├── setup_venv.bat             # Setup automat pentru Windows
├── run_app.bat                # Pornire rapidă aplicație web
├── requirements.txt            # Dependințe Python
├── setup_venv.bat             # Setup automat pentru Windows
├── README.md                  # Documentație (acest fișier)
│
└── venv/                      # Mediu virtual (se creează la instalare)
```

---

## Exemple de output

**Text original (150 cuvinte):**
```
Artificial intelligence is transforming the technology industry...
[text complet]
```

**Rezumat generat (35 cuvinte):**
```
AI is transforming tech industry. Major companies invest billions in research. 
Machine learning can now perform tasks requiring human intelligence. 
Experts predict AI will revolutionize healthcare, finance, and education.
```

**Compresie:** ~77%

---

## Concepte teoretice

### Extractive vs Abstractive Summarization

| Extractive | Abstractive |
|------------|-------------|
| Extrage propoziții din text | Generează text nou |
| Mai simplu | Mai complex |
| Poate părea tăiat | Mai natural, coerent |
| Exemplu: TF-IDF, TextRank | Exemplu: BART, T5, Pegasus |

### Modelul BART

BART = **B**idirectional and **A**uto-**R**egressive **T**ransformers

- **Encoder-Decoder** architecture
- Pre-antrenat pe CNN/DailyMail (news summarization)
- Combină avantajele BERT (encoder) și GPT (decoder)
- Excelent pentru:
  - Text summarization
  - Translation
  - Text generation

---

## Configurare lungime rezumat

În cod poți ajusta lungimea rezumatului modificând parametrii:

```python
configurari_lungime = {
    "scurt": {"min_length": 30, "max_length": 60},
    "mediu": {"min_length": 60, "max_length": 130},
    "lung": {"min_length": 100, "max_length": 200}
}
```

---

## Troubleshooting

**Eroare: "Python nu este instalat"**
- Descarcă Python de la: https://www.python.org/downloads/
- Asigură-te că este adăugat în PATH

**Eroare: "No module named 'transformers'"**
- Activează mediul virtual: `venv\Scripts\activate`
- Reinstalează: `pip install -r requirements.txt`

**Procesul durează mult la prima rulare**
- Normal! Modelul se descarcă (≈1.6 GB) la prima utilizare
- Se salvează în cache, următoarele rulări sunt rapide

**RAM insuficient**
- Modelul BART necesită ~4-6 GB RAM
- Închide alte aplicații
- Alternativă: folosește `facebook/bart-base` (mai mic)

---

## Resurse utile

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [facebook/bart-large-cnn Model](https://huggingface.co/facebook/bart-large-cnn)

---

## Autori

**Proiect realizat pentru:** Limbaje Formale - UTCN  
**Echipă:** 2 persoane  

---

## Licență

Acest proiect este realizat în scop educațional pentru UTCN.

---

## Extensii posibile

- [ ] Suport pentru limba română (model multilingv)
- [ ] Interfață web (Gradio/Streamlit)
- [ ] Deploy pe HuggingFace Spaces
- [ ] Comparație cu alte modele (T5, Pegasus)
- [ ] Fine-tuning pe dataset custom
- [ ] API REST pentru integrare

---

**Última actualizare:** Ianuarie 2026
