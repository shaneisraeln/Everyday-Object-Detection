"""UI theme, minimalist dark styling, SVG icons, and browser Web Speech integration scripts."""

# Reusable inline SVG icons (Lucide-inspired, minimalist, 0 emojis)
ICON_SCAN = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="3" x2="12" y2="7"/><line x1="12" y1="17" x2="12" y2="21"/></svg>"""

ICON_MIC = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>"""

ICON_SEND = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>"""

ICON_VOLUME = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>"""

ICON_STOP = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/></svg>"""

ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>"""

ICON_CAMERA = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>"""

ICON_CLEAR = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>"""

ICON_SPARK = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>"""

ICON_EYE = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>"""

CUSTOM_CSS = """
/* Minimalist Dark Obsidian Studio Theme */
:root {
    --bg-base: #09090b;
    --bg-card: #111115;
    --bg-surface: #17171c;
    --bg-input: #131317;
    --border-subtle: #23232a;
    --border-focus: #3f3f46;
    --border-hover: #2f2f38;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-emerald: #10b981;
    --accent-cyan: #06b6d4;
    --accent-danger: #ef4444;
}

body, .gradio-container {
    background-color: var(--bg-base) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif !important;
    color: var(--text-primary) !important;
    margin: 0 auto !important;
    max-width: 1400px !important;
    padding: 0 16px 32px 16px !important;
    -webkit-font-smoothing: antialiased;
}

/* Header Bar */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    margin-top: 12px;
    margin-bottom: 20px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: #18181f;
    border: 1px solid #272730;
    color: var(--accent-emerald);
}

.app-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.015em;
    margin: 0;
}

.app-meta {
    display: flex;
    align-items: center;
    gap: 8px;
}

.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    background: #18181f;
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.02em;
}

.meta-pill strong {
    color: var(--text-primary);
}

.status-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--accent-emerald);
    box-shadow: 0 0 8px var(--accent-emerald);
}

/* Studio Panels & Cards */
.panel-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
}

/* Inputs, Textareas & Selects */
input, textarea, select {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(63, 63, 70, 0.3) !important;
    outline: none !important;
}

/* Custom Primary Button */
button.btn-primary {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    background: #f4f4f5 !important;
    color: #09090b !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

button.btn-primary:hover {
    background: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.1) !important;
}

/* Secondary & Icon Buttons */
button.btn-secondary {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    background: #18181f !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

button.btn-secondary:hover {
    background: #23232b !important;
    border-color: var(--border-hover) !important;
}

/* Mic Button Pulse Effect */
.mic-btn-active {
    background-color: var(--accent-danger) !important;
    color: #ffffff !important;
    border-color: var(--accent-danger) !important;
    animation: mic-glow 1.4s infinite !important;
}

@keyframes mic-glow {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Target Object Card HUD */
.target-hud {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #141419;
    border: 1px solid #23232a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
}

.target-info {
    display: flex;
    align-items: center;
    gap: 10px;
}

.target-label {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #ffffff;
}

.target-conf {
    font-size: 0.78rem;
    font-family: monospace;
    color: var(--accent-emerald);
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 2px 7px;
    border-radius: 4px;
}

/* Status Bar */
.status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: monospace;
    padding: 6px 4px;
}

/* Answer Output Box */
.answer-box textarea {
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    background-color: #0e0e12 !important;
    color: #f1f1f4 !important;
    border: 1px solid #1f1f26 !important;
    border-radius: 8px !important;
}

/* Hide Gradio default label bloat */
.gr-form {
    border: none !important;
    background: transparent !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-base);
}
::-webkit-scrollbar-thumb {
    background: #27272f;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3f3f46;
}
"""

BROWSER_SPEECH_JS = """
<script>
window.isListening = false;
window.speechRecognitionInstance = null;

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("Web Speech API is not supported in this browser.");
        return null;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = function() {
        window.isListening = true;
        updateMicStatus("Listening... Speak clearly.");
        const btn = document.querySelector('#ptt_btn');
        if (btn) btn.classList.add('mic-btn-active');
    };

    recognition.onresult = function(event) {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        const inputElem = document.querySelector('#question_input textarea');
        if (inputElem) {
            inputElem.value = transcript;
            inputElem.dispatchEvent(new Event('input', { bubbles: true }));
        }
    };

    recognition.onerror = function(event) {
        window.isListening = false;
        console.error("Speech recognition error:", event.error);
        updateMicStatus("Speech recognition error: " + event.error);
        const btn = document.querySelector('#ptt_btn');
        if (btn) btn.classList.remove('mic-btn-active');
    };

    recognition.onend = function() {
        window.isListening = false;
        updateMicStatus("Microphone ready. Click mic or type below.");
        const btn = document.querySelector('#ptt_btn');
        if (btn) btn.classList.remove('mic-btn-active');
    };

    return recognition;
}

function updateMicStatus(statusText) {
    const statusElem = document.querySelector('#mic_status_display');
    if (statusElem) {
        statusElem.innerText = statusText;
    }
}

window.toggleVoiceRecognition = function() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Web Speech API is not supported in this browser. Please type your question directly.");
        updateMicStatus("Web Speech unsupported in this browser.");
        return;
    }
    if (!window.speechRecognitionInstance) {
        window.speechRecognitionInstance = initSpeechRecognition();
    }
    if (window.isListening) {
        window.speechRecognitionInstance.stop();
    } else {
        try {
            window.speechRecognitionInstance.start();
        } catch (e) {
            window.speechRecognitionInstance = initSpeechRecognition();
            window.speechRecognitionInstance.start();
        }
    }
};

window.speakCurrentAnswer = function() {
    if (!('speechSynthesis' in window)) {
        console.warn("Browser Text-to-Speech is not supported in this browser.");
        return;
    }
    window.speechSynthesis.cancel();
    const answerElem = document.querySelector('#answer_display textarea');
    const textToSpeak = answerElem ? answerElem.value : '';
    if (!textToSpeak || textToSpeak.trim() === '') {
        return;
    }
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
};

window.stopSpeaking = function() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
};
</script>
"""
