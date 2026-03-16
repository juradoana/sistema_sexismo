/**
 * Script para la interacción de la web 
 */
document.addEventListener('DOMContentLoaded', () => {
    // GESTIÓN DEL COMPONENTE DROPDOWN (menú desplegable)
    const customSelect = document.getElementById('CustomSelect');
    const selectTrigger = document.getElementById('SelectTrigger');
    const selectOptions = document.getElementById('SelectOptions');
    const hiddenInput = document.getElementById('StrategySelect');

    // apertura y cierre del menú al hacer click en el disparador (disparador)
    selectTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        customSelect.classList.toggle('open');
    });
    // Lógica de selección de opción
    selectOptions.querySelectorAll('.custom-select__option').forEach(option => {
        option.addEventListener('click', () => {
            // se sincroniza el valor seleccionado con el input oculto para el envío posterior
            hiddenInput.value = option.dataset.value;

            // se actualiza la interfaz visual (texto del trigger y estado de selección)
            selectTrigger.querySelector('.custom-select__text').textContent = option.dataset.label;

            // se actualiza el estado seleccionado 
            selectOptions.querySelectorAll('.custom-select__option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');

            // se cierra el menú tras la selección 
            customSelect.classList.remove('open');
        });
    });

    // cerrar el dropdown si el usuario hace click fuera del componente
    document.addEventListener('click', (e) => {
        if (!customSelect.contains(e.target)) {
            customSelect.classList.remove('open');
        }
    });

    // icono de ayuda y tooltip
    const helpBtn = document.getElementById('HelpBtn');
    const helpTooltip = document.getElementById('HelpTooltip');

    helpBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        helpTooltip.classList.toggle('visible');
        helpBtn.classList.toggle('active');
    });

    // cierre automático del tooltip al interactuar con otros elementos
    document.addEventListener('click', (e) => {
        if (!helpBtn.contains(e.target) && !helpTooltip.contains(e.target)) {
            helpTooltip.classList.remove('visible');
            helpBtn.classList.remove('active');
        }
    });

    // funcionalidad de "Auto-completado" al hacer click en un ejemplo se carga en el textarea
    helpTooltip.querySelectorAll('.help-tooltip__item').forEach(item => {
        item.addEventListener('click', () => {
            const text = item.textContent.trim();
            document.getElementById('TextInput').value = text;
            helpTooltip.classList.remove('visible');
            helpBtn.classList.remove('active');
            document.getElementById('TextInput').focus();
        });
    });

    // LÓGICA DE COMUNICACIÓN CON LA API 
    const analyzeBtn = document.getElementById('AnalyzeBtn');
    const textInput = document.getElementById('TextInput');
    const strategySelect = document.getElementById('StrategySelect');
    const resultSection = document.getElementById('ResultSection');
    const btnText = document.getElementById('BtnText');
    const btnSpinner = document.getElementById('BtnSpinner');
    const errorMessage = document.getElementById('ErrorMessage');

    // UI Elements for results
    const badgeLabel = document.getElementById('BadgeLabel');
    const confidenceBar = document.getElementById('ConfidenceBar');
    const confidenceValue = document.getElementById('ConfidenceValue');
    const sexistContent = document.getElementById('SexistContent');
    const nonSexistContent = document.getElementById('NonSexistContent');
    const explanationText = document.getElementById('ExplanationText');
    const counterNarrativeText = document.getElementById('CounterNarrativeText');

    //Listener principal para iniciar el análisis al hacer click en el botón "Analizar Texto"
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        const strategy = strategySelect.value;

        // validación básica de entrada antes de realizar la petición al servidor
        if (!text) {
            showError('Por favor, introduce un texto para analizar.');
            return;
        }

        // reseteo de estados visuales antes de la nueva carga
        hideError();
        resultSection.classList.add('hidden');
        setLoading(true);

        try {
            // petición asíncrona al endpoint de análisis (Flask/FastAPI/Node)
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, strategy })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error al procesar la solicitud.');
            }

            const data = await response.json();
            displayResults(data); // se procesan los datos recibidos

        } catch (error) {
            showError(error.message);
            console.error(error);
        } finally {
            // se restaura el botón independientemente del resultado
            setLoading(false);
        }
    });

    function displayResults(data) {
        resultSection.classList.remove('hidden');

        // cálculo y visualización de la barra de confianza (probabilidad) se muestra como porcentaje y se actualiza la barra de progreso
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceBar.style.width = `${confidencePercent}%`;

        // ajuste basado en la clasificación del modelo
        if (data.is_sexist) {
            // CASO sexista
            badgeLabel.textContent = "SEXISTA";
            badgeLabel.className = "badge sexist";
            confidenceBar.style.backgroundColor = "var(--danger-color)"; // Rojo para advertencia de sexismo

            sexistContent.classList.remove('hidden');
            nonSexistContent.classList.add('hidden');

            // pone la explicación y la contranarrativa generada
            explanationText.textContent = data.explanation;
            counterNarrativeText.textContent = data.counter_narrative;
        } else {
            // CASO No-sexista 
            badgeLabel.textContent = "NO SEXISTA";
            badgeLabel.className = "badge non-sexist";
            confidenceBar.style.backgroundColor = "var(--success-color)"; // Verde para indicar ausencia de sexismo

            sexistContent.classList.add('hidden');
            nonSexistContent.classList.remove('hidden');

            // mostrar explicación para no-sexista
            const nonSexistExplanationText = document.getElementById('NonSexistExplanationText');
            if (nonSexistExplanationText && data.explanation) {
                nonSexistExplanationText.textContent = data.explanation;
            }
        }
    }

    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = "Analizando...";
            btnSpinner.classList.remove('hidden');
        } else {
            btnText.textContent = "Analizar Texto";
            btnSpinner.classList.add('hidden');
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
