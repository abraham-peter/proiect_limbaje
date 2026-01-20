"""
Proiect Limbaje Formale - UTCN
Interfață Web pentru Text Summarization cu BART

Interfață Gradio simplă și intuitivă pentru summarization.
"""

import gradio as gr
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# INIȚIALIZARE MODEL (se încarcă o singură dată)
# ============================================================================

print("Se încarcă modelul BART...")
print("Primă rulare poate dura câteva minute...\n")

# Creează pipeline-ul global
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("Model încărcat! Interfața va porni în curând...\n")


# ============================================================================
# FUNCȚIA DE SUMMARIZATION PENTRU INTERFAȚĂ
# ============================================================================

def rezuma_simplu(text, config):
    """
    Rezumă un text folosind configurarea specificată.
    
    Args:
        text (str): Textul de rezumat
        config (dict): Configurare cu min_length și max_length
    
    Returns:
        str: Textul rezumat
    """
    rezultat = summarizer(
        text,
        max_length=config["max_length"],
        min_length=config["min_length"],
        do_sample=False,
        truncation=True
    )
    return rezultat[0]['summary_text']


def rezuma_hierarchical(text, config):
    """
    Rezumă un text lung folosind metoda hierarchică:
    1. Împarte textul în 2 jumătăți
    2. Rezumă fiecare jumătate separat
    3. Combină cele 2 rezumate într-un rezumat final
    
    Args:
        text (str): Textul lung de rezumat
        config (dict): Configurare cu min_length și max_length
    
    Returns:
        str: Rezumatul final
    """
    # Împarte textul în 2 jumătăți (aproximativ egale)
    cuvinte = text.split()
    mijloc = len(cuvinte) // 2
    
    jumatate1 = " ".join(cuvinte[:mijloc])
    jumatate2 = " ".join(cuvinte[mijloc:])
    
    # Rezumă fiecare jumătate (cu lungimi intermediate)
    config_intermediar = {
        "min_length": config["min_length"],
        "max_length": config["max_length"] + 50  # Puțin mai lung pentru intermediate
    }
    
    rezumat1 = rezuma_simplu(jumatate1, config_intermediar)
    rezumat2 = rezuma_simplu(jumatate2, config_intermediar)
    
    # Combină cele 2 rezumate
    rezumat_combinat = f"{rezumat1} {rezumat2}"
    
    # Rezumă combinația pentru rezultatul final
    rezumat_final = rezuma_simplu(rezumat_combinat, config)
    
    return rezumat_final


def detecteaza_limba(text):
    """
    Detectează simplă dacă textul pare să fie în română (conține caractere specifice).
    
    Args:
        text (str): Textul de verificat
    
    Returns:
        bool: True dacă pare română, False altfel
    """
    caractere_romanesti = ['ă', 'â', 'î', 'ș', 'ț', 'Ă', 'Â', 'Î', 'Ș', 'Ț']
    cuvinte_romanesti_comune = [
        'și', 'în', 'de', 'la', 'cu', 'că', 'într', 'această', 'pentru',
        'sunt', 'este', 'sau', 'mai', 'între', 'unul', 'asupra', 'către'
    ]
    
    # Verifică caractere speciale românești
    for char in caractere_romanesti:
        if char in text:
            return True
    
    # Verifică cuvinte comune românești
    text_lower = text.lower()
    count_cuvinte_ro = sum(1 for cuv in cuvinte_romanesti_comune if f' {cuv} ' in text_lower)
    if count_cuvinte_ro >= 3:
        return True
    
    return False


def rezuma_text_interfata(text, lungime, foloseste_hierarchical):
    """
    Funcție apelată de interfața Gradio pentru a rezuma textul.
    
    Args:
        text (str): Textul introdus de utilizator
        lungime (str): Lungimea dorită ("Scurt", "Mediu", "Lung")
        foloseste_hierarchical (bool): Dacă să folosească summarization hierarchical
    
    Returns:
        tuple: (rezumat, statistici)
    """
    # Validare input
    if not text or len(text.strip()) < 50:
        return "Te rog introdu un text mai lung (minim ~50 caractere).", ""
    
    # Verifică dacă textul este în română
    if detecteaza_limba(text):
        return """
**TEXT ÎN ROMÂNĂ DETECTAT!**

Modelul **facebook/bart-large-cnn** este antrenat **DOAR pe limba engleză**.
Text în română va cauza erori sau rezultate incoerente.

**Soluții:**
1. Traduce textul în engleză (folosește Google Translate)
2. Folosește un text în engleză

**Notă pentru proiect:** Există modele multilingve (mBART, mT5) care suportă română,
dar acest proiect demonstrează BART standard pentru engleză.
        """, ""
    
    # Configurări pentru diferite lungimi
    configurari = {
        "Scurt": {"min_length": 30, "max_length": 60},
        "Mediu": {"min_length": 60, "max_length": 130},
        "Lung": {"min_length": 100, "max_length": 200}
    }
    
    config = configurari.get(lungime, configurari["Mediu"])
    
    # Detectează dacă textul e foarte lung
    cuvinte_input = len(text.split())
    warning_msg = ""
    metoda_folosita = ""
    
    try:
        # Decide ce metodă să folosească
        if foloseste_hierarchical and cuvinte_input > 750:
            # Folosește metoda hierarchical pentru texte lungi
            warning_msg = f"\n**METODĂ:** Hierarchical Summarization (textul are {cuvinte_input} cuvinte)\n"
            metoda_folosita = "Hierarchical (2 pași)"
            rezumat = rezuma_hierarchical(text, config)
        else:
            # Folosește metoda standard
            if cuvinte_input > 750:
                warning_msg = f"\n**NOTĂ:** Textul are {cuvinte_input} cuvinte. Va fi trunchiat la ~750 cuvinte.\nActivează 'Hierarchical Summarization' pentru a procesa întreg textul.\n"
                metoda_folosita = "Standard (cu truncare)"
            else:
                metoda_folosita = "Standard"
            
            rezumat = rezuma_simplu(text, config)
        
        # Calculează statistici
        cuvinte_original = len(text.split())
        cuvinte_rezumat = len(rezumat.split())
        rata_compresie = (1 - cuvinte_rezumat / cuvinte_original) * 100
        
        # Formatează statisticile
        statistici = f"""
{warning_msg}
**STATISTICI:**
- **Cuvinte text original:** {cuvinte_original}
- **Cuvinte rezumat:** {cuvinte_rezumat}
- **Rata de compresie:** {rata_compresie:.1f}%
- **Metodă folosită:** {metoda_folosita}
        """
        
        return rezumat, statistici
        
    except Exception as e:
        error_msg = str(e)
        
        # Mesaj specific pentru erori comune
        if "index out of range" in error_msg.lower():
            return """
**EROARE: Text incompatibil cu modelul**

Acest model funcționează **DOAR cu text în limba engleză**.

**Cauze posibile:**
- Text în altă limbă (română, etc.)
- Caractere speciale nesuportate
- Text prea scurt sau prea lung

**Soluție:**
Folosește un text **în engleză** (vezi exemplele de mai jos).
            """, ""
        else:
            return f"Eroare: {error_msg}\n\nAsigură-te că folosești text în limba engleză.", ""


# ============================================================================
# TEXTE DE EXEMPLU (pentru butonul "Exemplu")
# ============================================================================

EXEMPLE = [
    [
        """Artificial intelligence is transforming the technology industry at an unprecedented pace. Major tech companies are investing billions of dollars into AI research and development. Machine learning models are now capable of performing tasks that were once thought to require human intelligence, such as image recognition, natural language processing, and strategic game playing. The breakthrough came with deep learning techniques and the availability of massive datasets. Experts predict that AI will continue to revolutionize sectors including healthcare, finance, transportation, and education in the coming years.""",
        "Mediu"
    ],
    [
        """Climate change represents one of the most pressing challenges facing humanity in the 21st century. Scientific evidence overwhelmingly demonstrates that global temperatures are rising due to increased concentrations of greenhouse gases in the atmosphere, primarily from burning fossil fuels and deforestation. The consequences are already visible: melting polar ice caps, rising sea levels, more frequent and severe weather events, and disruptions to ecosystems worldwide. Coastal cities face flooding risks, while agricultural regions experience droughts and unpredictable growing seasons. International efforts to combat climate change have led to agreements like the Paris Climate Accord, where nations committed to limiting global temperature increases to well below 2 degrees Celsius above pre-industrial levels. However, progress has been inconsistent, with some countries struggling to meet their emission reduction targets. Renewable energy technologies, including solar and wind power, have become increasingly cost-competitive with fossil fuels, offering hope for a transition to cleaner energy sources.""",
        "Lung"
    ],
    [
        """The exploration of Mars has captivated human imagination for decades, and recent technological advances have brought the possibility of human missions to the Red Planet closer to reality. NASA's Perseverance rover, which landed on Mars in February 2021, has been conducting groundbreaking research, including experiments to produce oxygen from the Martian atmosphere and searching for signs of ancient microbial life. Private companies like SpaceX are developing spacecraft specifically designed for Mars missions, with ambitious timelines for crewed flights. The challenges of sending humans to Mars are immense, including the long journey time, exposure to cosmic radiation, and the need to develop sustainable life support systems.""",
        "Scurt"
    ]
]


# ============================================================================
# CREAREA INTERFEȚEI GRADIO
# ============================================================================
# **în limba engleză**.
            
#             ⚠️ **IMPORTANT:** Modelul funcționează **DOAR cu texte în engleză**. Text în română sau alte limbi va da eroare!
def creeaza_interfata():
    """
    Creează și returnează interfața Gradio.
    """
    
    # CSS personalizat pentru interfață elegantă cu gradient
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    .gr-button {
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-input, .gr-textbox {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-input:focus, .gr-textbox:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    """
    
    with gr.Blocks(
        title="Text Summarization - BART",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interfata:
        
        # Header
        gr.Markdown(
            """
            # Text Summarization cu BART
            ### Proiect Limbaje Formale - UTCN
            
            Această aplicație folosește modelul **facebook/bart-large-cnn** pentru a genera rezumate 
            abstractive ale textelor în limba engleză.
            
            ---
            """
        )
        
        # Layout cu 2 coloane
        with gr.Row():
            # Coloana stânga - INPUT
            with gr.Column(scale=1):
                gr.Markdown("### Text Original")
                input_text = gr.Textbox(
                    label="Introdu textul (limba engleză)",
                    placeholder="Inserează aici textul pe care vrei să-l rezumi...\n\nExemplu: Climate change is one of the most pressing issues...\n\nNOTĂ: Pentru texte lungi (>750 cuvinte), activează 'Hierarchical Summarization'.",
                    lines=15,
                    max_lines=25
                )
                
                lungime = gr.Radio(
                    choices=["Scurt", "Mediu", "Lung"],
                    value="Mediu",
                    label="Lungimea rezumatului",
                    info="Alege cât de detaliat să fie rezumatul"
                )
                
                hierarchical = gr.Checkbox(
                    label="Hierarchical Summarization (pentru texte lungi >750 cuvinte)",
                    value=True,
                    info="Procesează texte lungi în 2 pași pentru rezultate mai bune"
                )
                
                with gr.Row():
                    btn_submit = gr.Button("Generează Rezumat", variant="primary", scale=2)
                    btn_clear = gr.Button("Șterge", scale=1)
            
            # Coloana dreapta - OUTPUT
            with gr.Column(scale=1):
                gr.Markdown("### Rezumat Generat")
                
                output_rezumat = gr.Textbox(
                    label="Rezumatul",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
                
                output_statistici = gr.Markdown(
                    label="Statistici"
                )
        
        # Exemple
        gr.Markdown("---")
        gr.Markdown("### Încearcă un exemplu:")
        
        gr.Examples(
            examples=EXEMPLE,
            inputs=[input_text, lungime],
            label="Click pe un exemplu pentru a-l încărca",
            cache_examples=False
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Model:** [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | 
            **Framework:** HuggingFace Transformers | 
            **Interfață:** Gradio
            """
        )
        
        # Event handlers
        btn_submit.click(
            fn=rezuma_text_interfata,
            inputs=[input_text, lungime, hierarchical],
            outputs=[output_rezumat, output_statistici]
        )
        
        btn_clear.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[input_text, output_rezumat, output_statistici]
        )
    
    return interfata


# ============================================================================
# PORNIRE APLICAȚIE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PORNIRE INTERFAȚĂ WEB")
    print("="*60 + "\n")
    
    app = creeaza_interfata()
    
    # Pornește serverul
    # share=True creează un link public temporar (opțional)
    # share=False => doar local
    app.launch(
        server_name="127.0.0.1",  # localhost
        server_port=7860,          # portul implicit Gradio
        share=False,               # schimbă în True pentru link public
        inbrowser=True             # deschide automat browserul
    )
