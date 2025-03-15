document.addEventListener('DOMContentLoaded', () => {
    const savingsForm = document.getElementById('savingsForm');
    const savingsOutput = document.getElementById('savingsOutput');

    savingsForm.addEventListener('submit', (event) => {
        event.preventDefault();

        const currentUsage = parseFloat(document.getElementById('currentUsage').value);
        const potentialUsage = parseFloat(document.getElementById('potentialUsage').value);
        const rate = parseFloat(document.getElementById('rate').value);
        
        if (isNaN(currentUsage) || currentUsage < 0 || isNaN(potentialUsage) || potentialUsage < 0 || isNaN(rate) || rate < 0) {
            savingsOutput.innerHTML = '<p style="color: red;">Please enter valid numbers for all fields.</p>';
            return;
        }

        // Calculate savings
        const dailySavings = (currentUsage - potentialUsage) * rate;
        const annualSavings = dailySavings * 365;
        
        savingsOutput.innerHTML = `
            <p>Daily Savings: $${dailySavings.toFixed(2)}</p>
            <p>Estimated Annual Savings: $${annualSavings.toFixed(2)}</p>
        `;
    });
});
