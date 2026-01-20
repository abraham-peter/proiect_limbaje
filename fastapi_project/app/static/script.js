document.getElementById('inputText').addEventListener('input', function() {
    document.getElementById('charCount').textContent = this.value.length + " characters";
});

async function summarizeText() {
    const text = document.getElementById('inputText').value;
    const outputDiv = document.getElementById('summaryOutput');
    const loadingDiv = document.getElementById('loading');
    
    if (!text.trim()) {
        alert("Please enter some text to summarize.");
        return;
    }

    // UI Updates
    loadingDiv.classList.remove('hidden');
    outputDiv.classList.add('hidden');
    
    try {
        const formData = new FormData();
        formData.append('text', text);

        const response = await fetch('/summarize', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        outputDiv.innerHTML = data.summary;
    } catch (error) {
        outputDiv.innerHTML = "An error occurred: " + error.message;
    } finally {
        loadingDiv.classList.add('hidden');
        outputDiv.classList.remove('hidden');
    }
}
