"""
Proiect Limbaje Formale - UTCN
Text Summarization folosind BART (facebook/bart-large-cnn)

Acest script demonstrează Abstractive Summarization folosind 
transformer-ul BART pre-antrenat pe dataset-ul CNN/DailyMail.
"""

from transformers import pipeline
import warnings

# Ignoră warnings pentru output mai curat
warnings.filterwarnings('ignore')


# ============================================================================
# TEXTE DE EXEMPLU (Articole în engleză de diferite lungimi)
# ============================================================================

TEXTE_EXEMPLU = {
    "scurt": """
    Artificial intelligence is transforming the technology industry at an unprecedented pace. 
    Major tech companies are investing billions of dollars into AI research and development. 
    Machine learning models are now capable of performing tasks that were once thought to 
    require human intelligence, such as image recognition, natural language processing, and 
    strategic game playing. The breakthrough came with deep learning techniques and the 
    availability of massive datasets. Experts predict that AI will continue to revolutionize 
    sectors including healthcare, finance, transportation, and education in the coming years.
    """,
    
    "mediu": """
    Climate change represents one of the most pressing challenges facing humanity in the 
    21st century. Scientific evidence overwhelmingly demonstrates that global temperatures 
    are rising due to increased concentrations of greenhouse gases in the atmosphere, 
    primarily from burning fossil fuels and deforestation. The consequences are already 
    visible: melting polar ice caps, rising sea levels, more frequent and severe weather 
    events, and disruptions to ecosystems worldwide. Coastal cities face flooding risks, 
    while agricultural regions experience droughts and unpredictable growing seasons. 
    
    International efforts to combat climate change have led to agreements like the Paris 
    Climate Accord, where nations committed to limiting global temperature increases to 
    well below 2 degrees Celsius above pre-industrial levels. However, progress has been 
    inconsistent, with some countries struggling to meet their emission reduction targets. 
    
    Renewable energy technologies, including solar and wind power, have become increasingly 
    cost-competitive with fossil fuels, offering hope for a transition to cleaner energy 
    sources. Electric vehicles are gaining market share, and governments are implementing 
    policies to phase out internal combustion engines. Scientists emphasize that immediate 
    and substantial action is required to prevent the most catastrophic effects of climate 
    change and preserve the planet for future generations.
    """,
    
    "lung": """
    The advent of quantum computing promises to revolutionize the computational landscape 
    in ways that were previously confined to the realm of science fiction. Unlike classical 
    computers that use bits representing either 0 or 1, quantum computers utilize quantum 
    bits, or qubits, which can exist in multiple states simultaneously through a phenomenon 
    called superposition. This fundamental difference allows quantum computers to process 
    vast amounts of information in parallel, potentially solving certain problems 
    exponentially faster than the most powerful classical supercomputers.
    
    The development of quantum computing has been driven by advances in quantum mechanics, 
    materials science, and cryogenic engineering. Major technology companies and research 
    institutions worldwide are racing to build practical quantum computers. IBM, Google, 
    Microsoft, and numerous startups have made significant progress, with Google claiming 
    to have achieved "quantum supremacy" in 2019 when their quantum processor performed 
    a specific calculation that would take classical computers thousands of years.
    
    Potential applications of quantum computing span numerous fields. In cryptography, 
    quantum computers could break many current encryption schemes, necessitating the 
    development of quantum-resistant cryptographic methods. In drug discovery, they could 
    simulate molecular interactions at unprecedented scales, accelerating the development 
    of new medications. Financial institutions are exploring quantum algorithms for 
    portfolio optimization and risk analysis. Climate scientists hope to use quantum 
    simulations to create more accurate models of atmospheric processes.
    
    However, significant challenges remain before quantum computers become practical for 
    widespread use. Qubits are extremely fragile and susceptible to environmental 
    interference, a problem known as decoherence. Maintaining qubits requires cooling 
    systems that operate near absolute zero temperature, making quantum computers 
    expensive and difficult to maintain. Error correction in quantum systems is far more 
    complex than in classical computing, requiring sophisticated algorithms and additional 
    qubits dedicated to error detection and correction.
    
    Despite these obstacles, researchers remain optimistic about the future of quantum 
    computing. Incremental improvements in qubit stability, error correction techniques, 
    and quantum algorithms continue to emerge. Some experts predict that within the next 
    decade, quantum computers will begin to solve real-world problems that are intractable 
    for classical computers, ushering in a new era of computational capability that could 
    transform industries, accelerate scientific discovery, and reshape our understanding 
    of computation itself.
    """,
    
    "custom": """
    The exploration of Mars has captivated human imagination for decades, and recent 
    technological advances have brought the possibility of human missions to the Red 
    Planet closer to reality. NASA's Perseverance rover, which landed on Mars in 
    February 2021, has been conducting groundbreaking research, including experiments 
    to produce oxygen from the Martian atmosphere and searching for signs of ancient 
    microbial life. Private companies like SpaceX are developing spacecraft specifically 
    designed for Mars missions, with ambitious timelines for crewed flights.
    
    The challenges of sending humans to Mars are immense. The journey would take 
    approximately six to nine months each way, exposing astronauts to cosmic radiation 
    and the psychological effects of isolation. Mars has only 38% of Earth's gravity, 
    which could cause muscle atrophy and bone density loss during extended stays. The 
    thin atmosphere, composed mainly of carbon dioxide, provides little protection from 
    radiation and makes landing large spacecraft extremely difficult.
    
    Establishing a sustainable human presence on Mars would require developing life 
    support systems, habitats that protect against radiation and extreme temperatures, 
    and methods for producing food, water, and fuel from local resources. Some 
    scientists propose using subsurface lava tubes as ready-made shelters, while others 
    envision 3D-printed habitats constructed from Martian soil.
    """
}


# ============================================================================
# FUNCȚIA PRINCIPALĂ DE SUMMARIZATION
# ============================================================================

def creeaza_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Creează și returnează un pipeline de summarization folosind modelul specificat.
    
    Args:
        model_name (str): Numele modelului HuggingFace (default: facebook/bart-large-cnn)
    
    Returns:
        Pipeline object pentru summarization
    """
    print(f"📥 Se încarcă modelul {model_name}...")
    print("⏳ Aceasta poate dura câteva secunde la prima rulare...\n")
    
    # Creează pipeline-ul de summarization
    # task="summarization" specifică că vrem să rezumăm text
    summarizer = pipeline("summarization", model=model_name)
    
    print("✅ Model încărcat cu succes!\n")
    return summarizer


def rezuma_text(summarizer, text, lungime="mediu"):
    """
    Rezumă un text folosind modelul BART.
    
    Args:
        summarizer: Pipeline-ul de summarization
        text (str): Textul de rezumat
        lungime (str): Lungimea rezumatului ("scurt", "mediu", "lung")
    
    Returns:
        str: Textul rezumat
    """
    # Configurări pentru diferite lungimi de rezumat
    # min_length și max_length controlează dimensiunea output-ului
    configurari_lungime = {
        "scurt": {"min_length": 30, "max_length": 60},
        "mediu": {"min_length": 60, "max_length": 130},
        "lung": {"min_length": 100, "max_length": 200}
    }
    
    # Obține configurarea pentru lungimea dorită
    config = configurari_lungime.get(lungime, configurari_lungime["mediu"])
    
    print(f"🔄 Generare rezumat (lungime: {lungime})...")
    
    # Generează rezumatul
    # do_sample=False => folosește greedy decoding (deterministă)
    # truncation=True => taie textul dacă e prea lung pentru model
    rezultat = summarizer(
        text,
        max_length=config["max_length"],
        min_length=config["min_length"],
        do_sample=False,
        truncation=True
    )
    
    # Extrage textul rezumat din rezultat
    return rezultat[0]['summary_text']


def afiseaza_comparatie(text_original, rezumat, tip_text=""):
    """
    Afișează o comparație vizuală între textul original și rezumat.
    
    Args:
        text_original (str): Textul original
        rezumat (str): Textul rezumat
        tip_text (str): Tipul textului (pentru titlu)
    """
    # Calculează statistici
    cuvinte_original = len(text_original.split())
    cuvinte_rezumat = len(rezumat.split())
    rata_compresie = (1 - cuvinte_rezumat / cuvinte_original) * 100
    
    print("=" * 80)
    if tip_text:
        print(f"📄 TIP TEXT: {tip_text.upper()}")
    print("=" * 80)
    
    print(f"\n📊 STATISTICI:")
    print(f"   • Cuvinte text original: {cuvinte_original}")
    print(f"   • Cuvinte rezumat: {cuvinte_rezumat}")
    print(f"   • Rata de compresie: {rata_compresie:.1f}%")
    
    print(f"\n📝 TEXT ORIGINAL ({cuvinte_original} cuvinte):")
    print("-" * 80)
    print(text_original.strip())
    
    print(f"\n✨ REZUMAT ({cuvinte_rezumat} cuvinte):")
    print("-" * 80)
    print(rezumat)
    print("\n")


# ============================================================================
# FUNCȚIA MAIN - DEMONSTRAȚIE
# ============================================================================

def main():
    """
    Funcția principală care demonstrează summarization pe textele de exemplu.
    """
    print("\n" + "="*80)
    print("🚀 PROIECT LIMBAJE FORMALE - TEXT SUMMARIZATION cu BART")
    print("="*80 + "\n")
    
    # Creează summarizer-ul
    summarizer = creeaza_summarizer()
    
    # -------------------------------------------------------------------------
    # DEMO 1: Text scurt
    # -------------------------------------------------------------------------
    print("\n" + "🔵"*40)
    print("DEMONSTRAȚIE 1: TEXT SCURT")
    print("🔵"*40 + "\n")
    
    text_scurt = TEXTE_EXEMPLU["scurt"]
    rezumat_scurt = rezuma_text(summarizer, text_scurt, lungime="scurt")
    afiseaza_comparatie(text_scurt, rezumat_scurt, tip_text="Scurt - AI Industry")
    
    
    # -------------------------------------------------------------------------
    # DEMO 2: Text mediu
    # -------------------------------------------------------------------------
    print("\n" + "🟢"*40)
    print("DEMONSTRAȚIE 2: TEXT MEDIU")
    print("🟢"*40 + "\n")
    
    text_mediu = TEXTE_EXEMPLU["mediu"]
    rezumat_mediu = rezuma_text(summarizer, text_mediu, lungime="mediu")
    afiseaza_comparatie(text_mediu, rezumat_mediu, tip_text="Mediu - Climate Change")
    
    
    # -------------------------------------------------------------------------
    # DEMO 3: Text lung
    # -------------------------------------------------------------------------
    print("\n" + "🟡"*40)
    print("DEMONSTRAȚIE 3: TEXT LUNG")
    print("🟡"*40 + "\n")
    
    text_lung = TEXTE_EXEMPLU["lung"]
    rezumat_lung = rezuma_text(summarizer, text_lung, lungime="lung")
    afiseaza_comparatie(text_lung, rezumat_lung, tip_text="Lung - Quantum Computing")
    
    
    # -------------------------------------------------------------------------
    # DEMO 4: Comparație diferite lungimi pe același text
    # -------------------------------------------------------------------------
    print("\n" + "🔴"*40)
    print("DEMONSTRAȚIE 4: ACELAȘI TEXT - DIFERITE LUNGIMI")
    print("🔴"*40 + "\n")
    
    text_custom = TEXTE_EXEMPLU["custom"]
    
    print("📌 Rezumat SCURT:")
    print("-" * 80)
    rezumat_custom_scurt = rezuma_text(summarizer, text_custom, lungime="scurt")
    print(rezumat_custom_scurt)
    print()
    
    print("📌 Rezumat MEDIU:")
    print("-" * 80)
    rezumat_custom_mediu = rezuma_text(summarizer, text_custom, lungime="mediu")
    print(rezumat_custom_mediu)
    print()
    
    print("📌 Rezumat LUNG:")
    print("-" * 80)
    rezumat_custom_lung = rezuma_text(summarizer, text_custom, lungime="lung")
    print(rezumat_custom_lung)
    print()
    
    
    # -------------------------------------------------------------------------
    # SECȚIUNE INTERACTIVĂ (opțional)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("💡 OPȚIUNE: Vrei să testezi cu propriul tău text?")
    print("="*80)
    print("Decomentează secțiunea interactivă din cod sau")
    print("folosește funcția rezuma_text() direct cu textul tău.\n")
    
    # Decomentează liniile de mai jos pentru input interactiv:
    """
    print("\n📝 Introdu textul tău (apasă Enter de 2 ori când termini):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    text_utilizator = "\n".join(lines)
    
    if text_utilizator.strip():
        print("\nAlege lungimea rezumatului: scurt / mediu / lung")
        lungime_aleasa = input("Lungime: ").strip().lower()
        if lungime_aleasa not in ["scurt", "mediu", "lung"]:
            lungime_aleasa = "mediu"
        
        rezumat_utilizator = rezuma_text(summarizer, text_utilizator, lungime=lungime_aleasa)
        afiseaza_comparatie(text_utilizator, rezumat_utilizator, tip_text="Text Utilizator")
    """
    
    print("\n" + "="*80)
    print("✅ DEMONSTRAȚIE COMPLETĂ!")
    print("="*80 + "\n")


# ============================================================================
# PUNCT DE INTRARE
# ============================================================================

if __name__ == "__main__":
    main()
