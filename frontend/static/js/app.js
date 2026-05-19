// Lógica principal del Frontend PDA

document.addEventListener('DOMContentLoaded', () => {
    // 1. Cargar estado inicial (fuentes en memoria)
    updateSourcesCount();
});

// Función para actualizar el contador del Header
async function updateSourcesCount() {
    try {
        const response = await fetch('/api/sources');
        const data = await response.json();
        const count = data.sources ? data.sources.length : 0;
        
        const badge = document.getElementById('sources-count-badge');
        if (badge) {
            badge.textContent = `Sources: ${count}`;
        }
    } catch (error) {
        console.error("Error fetching sources count:", error);
    }
}

// Utilidad para notificaciones (Toasts) simple
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded shadow-lg z-50 text-white font-body-md transition-opacity duration-300`;
    
    if (type === 'success') {
        toast.classList.add('bg-primary', 'text-[#0f172a]', 'font-bold');
    } else if (type === 'error') {
        toast.classList.add('bg-error');
    } else {
        toast.classList.add('bg-surface-container-high', 'border', 'border-outline-variant');
    }
    
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
